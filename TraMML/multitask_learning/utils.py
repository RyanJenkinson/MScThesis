import logging

import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import TensorDataset, DataLoader

from extract_features import convert_examples_to_features
from modelling import MultiTaskModel, TOKENIZERS

LOGGER = logging.getLogger(__name__)

def flatten_list(lst):
    """
    Given a (generic) list of lists, flatten it into a single list
    If lst is a regular list then flattened_list = lst but if it is a list of lists then it gets flattened
    This is used for the NER task when the label is a list of lists, but we just
    want to return the unique labels with this function

    Parameters
    ----------
    lst : list of lists
        A list of list of labels (could just be a single list of labels)
    """
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(item)
        else:
            flattened_list.append(item)
    return flattened_list

class AspectBasedDataset:
    """
    AspectBasedDataset is a class that, given a set of responses and Streetbees-style multi-labels "['Happy', 'Sad']",
    creates an aspect-based dataset. This is essentially a dataframe in the form:

                            ......................................................
                            id      label   text_a              text_b
                            0       0       I feel sad          Do you feel happy?
                            1       1       I feel sad          Do you feel sad?
                            2       1       I am happy today    Do you feel happy?
                            3       0       I am happy today    Do you feel sad?
                            ......................................................


    where id is a simple unique id, label is if the statement is true in text_b, and text_a is the set of sentences
    we wish to learn from (the responses).

    """
    def __init__(self, responses, labels_strings=None, threshold=100, test_size=0.1, desparsify_num_neg_samples=5, prepend_question_string='Do you feel'):
        """
        Initialise AspectBasedDataset object.

        Parameters
        ----------
        responses : list
            list of responses to be analysed
        labels_strings : list
            list of multi-label strings to be analysed, by default None
        threshold : int
            minimum occurrence rate for a label to be considered. Default 100
        test_size : float
            size of test set
        desparsify_num_neg_samples : int or None, optional
            Since we make our dataset with Q questions into C*Q entries (where C is the
            number of classes) we are expanding our dataset massively, to counteract this,
            we may choose to keep all positive samples, and randomly sample num_negative_samples
            negative samples in an attempt to slightly balance the classes. If None, no desparsification.
            NOTE: We only want to desparsify on the training data, never on the val/testing data
            If int, it is the number of negative samples to keep when desparsifying, by default 5
        prepend_question_string : str
            The string to prepend the question by, so that each question is of the form:
            prepend question string + aspect category + ? e.g 'Do you feel <emotion category>?'

        """
        if labels_strings is not None:
            included_labels = AspectBasedDataset.calculate_threshold_labels(labels_strings, threshold)
            #TODO: Make this a train_dev_test split!
            train_responses, test_responses, train_labels, test_labels = train_test_split(responses,
                                                                                          labels_strings,
                                                                                          test_size=test_size,
                                                                                          random_state=42)

            self.train = AspectBasedDataset.create_dataset(included_labels, train_responses, train_labels,
                                                           desparsify_num_neg_samples, prepend_question_string)
            self.test = AspectBasedDataset.create_dataset(included_labels, test_responses, test_labels,
                                                          desparsify_num_neg_samples=None,
                                                          prepend_question_string=prepend_question_string)
        else:
            self.aspect_categories = ['Mindful', 'Motivated', 'Neutral', 'Anxious / worried', 'Other', 'Energetic',
                                      'Concerned', 'Cool', 'Healthy', 'Positive', 'Relaxed', 'Busy', 'Thankful',
                                      'Craving', 'Cold', 'Sleepy / Tired', 'Hot', 'Full', 'Thirsty', 'Rested / Awake',
                                      'Good', 'Peaceful', 'Refreshed', 'Lazy', 'Bored', 'Annoyed / angry',
                                      'Sad / depressed', 'Thoughtful', 'Great / amazing', 'Hungry', 'Unwell', 'Excited',
                                      'Happy / content', 'Fine']
            self.test = AspectBasedDataset.create_dataset(included_labels=self.aspect_categories, responses=responses,
                                                          labels_strings=None)


    @staticmethod
    def labels_to_list(labels):
        """
        Convert labels in a string format to actual python list. 'Other' seems to be in a different format, so an
        exception is made for it.

        Parameters
        ----------
        labels : string
            labels in a string format

        Returns
        -------
        list
            converted list of labels
        """
        if labels == 'Other':
            return ['Other']
        else:
            return ast.literal_eval(labels)

    @staticmethod
    def calculate_threshold_labels(labels_strings, threshold):
        """
        Find labels above the threshold frequency.

        Parameters
        ----------
        labels_strings : list of strings
            labels in string multi-class format (Standard Streebees)
        threshold : int
            minimum occurrence rate for a label to be considered

        Returns
        -------
        included_labels : list
            list of labels to include
        """

        converted_labels = []
        for labels in labels_strings:
            labels = AspectBasedDataset.labels_to_list(labels)

            for label in labels:
                    converted_labels.append(label)

        series_labels = pd.Series(converted_labels)
        value_counts = series_labels.value_counts()
        excluded_labels = list(value_counts[value_counts <= threshold].index)

        unique_labels = list(set(converted_labels))
        included_labels = []
        for label in unique_labels:
            if label not in excluded_labels:
                included_labels.append(label)

        return included_labels

    @staticmethod
    def create_dataset(included_labels, responses, labels_strings=None, desparsify_num_neg_samples=5, prepend_question_string='Do you feel'):
        """
        Find labels above the threshold frequency.

        Parameters
        ----------
        responses : list
            list of responses to be analysed
        labels_strings : list
            list of multi-label strings to be analysed, by default none
        included_labels : list
            list of labels to include
        desparsify_num_neg_samples : int or None, optional
            Since we make our dataset with Q questions into C*Q entries (where C is the
            number of classes) we are expanding our dataset massively, to counteract this,
            we may choose to keep all positive samples, and randomly sample num_negative_samples
            negative samples in an attempt to slightly balance the classes. If None, no desparsification.
            NOTE: We only want to desparsify on the training data, never on the val/testing data
            If int, it is the number of negative samples to keep when desparsifying, by default 5
        prepend_question_string : str
            The string to prepend the question by, so that each question is of the form:
            prepend question string + aspect category + ? e.g 'Do you feel <emotion category>?'

        Returns
        -------
        dataframe
            aspect-based dataframe
        """

        def clean_text(text):
            """
            A simple function to remove emojis and \n tags etc

            Parameters
            ----------
            text : str
                Input text to be cleaned

            Returns
            -------
            str
                The cleaned text
            """
            cleaned_text = text.encode('ascii', 'ignore').decode('ascii')

            strings_to_remove = ['\n', '\r', '\t']
            for string_to_remove in strings_to_remove:
                cleaned_text = cleaned_text.replace(string_to_remove, '')
            return cleaned_text
        if labels_strings is not None:
            text_a = []
            text_b = []
            feeling_match = []
            for labels, response in zip(labels_strings, responses):
                labels = AspectBasedDataset.labels_to_list(labels)
                response = clean_text(response)
                if response:  # If after we have cleaned the response it is non empty then continue
                    for label in labels:
                        for included_label in included_labels:
                            text_a.append(response)
                            text_b.append(' '.join([prepend_question_string, included_label, '?']))
                            if included_label == label:
                                feeling_match.append(1)
                            else:
                                feeling_match.append(0)

            ids = list(range(len(feeling_match)))
            d = {'id': ids, 'label': feeling_match, 'text_a': text_a, 'text_b': text_b}
            df = pd.DataFrame(data=d)

            if desparsify_num_neg_samples:
                negative_sample_fn = lambda df: df.loc[np.random.choice(df.index, size=desparsify_num_neg_samples, replace=False), :]
                df = pd.concat([df[df['label'] == 1],
                                df[df['label'] == 0].groupby('text_a', as_index=False).apply(negative_sample_fn)]).sample(frac=1)
        else:
            text_a = []
            text_b = []
            aspect_categories = included_labels
            for response in responses:
                response = clean_text(response)
                if response:
                    for aspect_category in aspect_categories:
                        text_a.append(response)
                        text_b.append(' '.join([prepend_question_string, included_label, '?']))
            ids = list(range(len(text_a)))
            df = pd.DataFrame(data={'id': ids, 'text_a': text_a, 'text_b': text_b})
        return df.set_index('id')

    def save_train_test(self, filename='dataset'):
        """
        save train_test to csv

        Parameters
        ----------
        filename : string
            name of files to be saved
        """
        self.train.to_csv(filename+'_train.csv')
        self.test.to_csv(filename+'_test.csv')

class DatasetPredictor:
    """
    docstring
    """
    def __init__(self, path_to_test_folder, path_to_pretrained_model_ckpt, trained_model_name='bert-base-cased'):
        """
        Initialises the dataset predictor class

        Parameters
        ----------
        path_to_test_folder : str
            Path to the folder with the test data (possibly could be made redundant in init and passed into make_predictions only)
            Currently, it needs to be added for the processor object
        path_to_pretrained_model_ckpt : str
            Path to the pretrained model checkpoint
        trained_model_name : str, optional
            Name of the pretrained model, by default 'bert-base-cased'
        """
        self.data_dir = path_to_test_folder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Might want to abstract away the processor for future tasks, or move processor to init from make_predictions
        # processor = ... [e.g StreetbeesMoodProcessor(data_dir, labelled_data=...)]
        # self.task_name = processor.data_name
        # task_configs = {self.task_name: {"num_labels": processor.num_labels,
        #                                  "task_type": processor.task_type,
        #                                  "output_type": processor.output_type}
        task_configs = {'Streetbees_Mood': {"num_labels": 2,
                                            "task_type": 'Primary',
                                            "output_type": 'CLS'}}
        self.model = MultiTaskModel(task_configs=task_configs,
                                    model_name_or_config='bert-base-cased')
        pretrained_model_dict = torch.load(path_to_pretrained_model_ckpt, map_location=self.device)
        pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(pretrained_model_dict)
        self.model.to(self.device)

        # Initialise the tokenizer based on the model fed in
        do_lower_case = False if trained_model_name.split("-")[-1] == "cased" else True
        self.baseLM_model_name = trained_model_name.split("-")[0].lower()
        self.tokenizer = TOKENIZERS[self.baseLM_model_name].from_pretrained(trained_model_name,
                                                                       do_lower_case=do_lower_case)

    def make_predictions(self, input_df=None, labelled_data=True, data_processor=None):
        """
        Make predictions using the Language Model

        Parameters
        ----------
        input_df : pandas dataframe, optional
            If inputted, this is the dataframe that we will make predictions from. If not given, then
            the system will automatically look in the processed_data folder in datascience-projects and get the
            test data corresponding to the folder defined by the processor object, by default None
        labelled_data : bool, optional
            Indicates whether or not the data is labelled, by default True
        data_processor : DataProcessor object
            A data processor object from data_processing.py that processes the data. We use this for Streetbees Mood
            data to get the test examples, the name and the labels etc

        Returns
        -------
        array or tuple
            If labelled_data: tuple (y_preds, test_accuracy)
            if not labelled data: array y_preds
        """
        processor = data_processor(data_dir=self.data_dir, labelled_data=labelled_data)
        task_name = processor.data_name
        label_list = processor.get_labels()
        test_examples = processor.get_examples(input_df=input_df, set_type="test")
        max_seq_length = 128  # All models trained with this max_seq_length

        # Load in the testing data
        test_features = convert_examples_to_features(test_examples, label_list, max_seq_length, self.tokenizer, task_name, self.baseLM_model_name)
        all_input_ids = torch.tensor([feature.input_ids for feature in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([feature.segment_ids for feature in test_features], dtype=torch.long)
        all_input_masks = torch.tensor([feature.input_mask for feature in test_features], dtype=torch.long)
        if labelled_data:
            all_label_ids = torch.tensor([feature.label_id for feature in test_features], dtype=torch.long)
            test_data = TensorDataset(all_input_ids, all_segment_ids, all_input_masks, all_label_ids)
        else:
            test_data = TensorDataset(all_input_ids, all_segment_ids, all_input_masks)

        test_dataloader = DataLoader(test_data)
        # Calculate the accuracies, losses (if labelled_data) and return the predictions
        y_pred = []
        if labelled_data:
            y_true = []
            test_total_correct = 0

        # Evaluate the model with batchwise accuracy like in training
        for batch in tqdm(test_dataloader, desc="Getting Predictions"):
            if labelled_data:
                input_ids, segment_ids, input_masks, label_ids = tuple(model_input.to(self.device) for model_input in batch)
                with torch.no_grad():
                    _, logits = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)
            else:
                input_ids, segment_ids, input_masks = tuple(model_input.to(self.device) for model_input in batch)
                with torch.no_grad():
                    logits = self.model(input_ids, segment_ids, input_masks, task_name)

            logits = logits.detach().cpu().numpy()
            y_pred_batch = np.argmax(logits, axis=1)
            y_pred = np.concatenate([y_pred, y_pred_batch])
            if labelled_data:
                y_true_batch = label_ids.to('cpu').numpy()
                y_true = np.concatenate([y_true, y_true_batch])
                batch_num_correct = np.sum(y_pred_batch == y_true_batch)
                test_total_correct += batch_num_correct

        if labelled_data:
            test_accuracy = test_total_correct / len(test_data)
            LOGGER.info('\n' + classification_report(y_true, y_pred, target_names=label_list))
            return y_pred, test_accuracy
        else:
            return y_pred
