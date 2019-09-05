# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch BERT model. The following code is inspired by the HuggingFace
version and it is built upon in places (license below). In particular it
can be used and modified for commercial use:
https://github.com/huggingface/pytorch-pretrained-BERT/
In this code we create data preprocessing pipelines for the data.
"""
import os
import ast

import pandas as pd

from extract_features import InputExample
from utils import flatten_list

TASK_PRIORITIES = {'SST-2': 'Secondary', 'SST-5': 'Secondary',
                   'IMDB': 'Secondary', 'SemEval_QA_B': 'Primary',
                   'SemEval_QA_M': 'Primary', 'Streetbees_Mood': 'Primary',
                   'NER': 'Secondary', 'Sentihood_QA_B': 'Primary'}

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input.

    Parameters
    ----------
    text : str
        text to be converted to unicode

    Returns
    -------
    str
        converted text

    Raises
    ------
    ValueError
        if unsupported stringtype
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    elif isinstance(text, int) or isinstance(text,float):
        return str(text).encode("utf-8").decode("utf-8")
    elif isinstance(text, list):
        return [convert_to_unicode(item) for item in text]
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class DataProcessor(object):
    """
    Base class for data converters for sequence classification data sets.
    """
    def __init__(self, data_dir, data_name, task_type='infer', output_type="CLS", labelled_data=True, labels=None):
        """
        Initialisation for all DataProcessor Objects

        Parameters
        ----------
        data_dir : str
            Directory for the data in the standard format
        data_name : str
            Name of the processor
        task_type : str
            The `type` of task based on importance. So far one of ['Primary', 'Secondary', 'Tertiary']. Can either be inferred from TASK_PRIORITIES
            or explicitly given as a string, by default 'infer'
        output_type : str
            Either "CLS" for classification or "REG" for regression, default "CLS"
        labelled_data : bool
            Indicates whether or not our processor will be dealing with unlabelled (test) data e.g live
            Streetbees data. When running experiments, we always have labelled data from which we can calculate
            the accuracy etc but this flag can be used for the productionised models. by default, True, since we
            are primarily running experiments
        labels : list
            If not labelled_data, give the required labels that should be outputted
        """
        self.data_dir = data_dir
        self.data_name = data_name
        self.labelled_data = labelled_data
        self.labels = labels
        self.num_labels = len(self.get_labels())
        self.task_type = task_type if task_type != 'infer' else TASK_PRIORITIES[data_name]
        assert self.task_type in ['Primary', 'Secondary', 'Tertiary']
        self.output_type = output_type

    def get_examples(self, input_df=None, set_type='test'):
        """Gets a collection of `InputExample`s for a given dataset

        Parameters
        ----------
        input_df : pandas dataframe
            The input dataframe from which to create examples (usually put in directly at test time)
        set_type : str, optional
            A string indicating the type of set our data comes from
            Should be in ['train','dev','test'], by default 'test'

        Returns
        -------
        list
            list of InputExample's for the given dataset
        """
        if input_df is None:
            file_path_or_df = os.path.join(self.data_dir, self.data_name + "_" + set_type + ".csv")
        else:
            file_path_or_df = input_df
        texts_a, texts_b, labels = self._read_csv_or_df(file_path_or_df)
        return self._create_examples(texts_a, texts_b, labels, set_type=set_type)

    def get_labels(self):
        """Gets the list of unique labels for this dataset

        Returns
        -------
        label_list : list of strings
            Sorted (alphabetical/numerical) list of labels as strings
        """
        if self.labels or not self.labelled_data:
            return self.labels
        else:
            all_labels = []
            # Could add in _test.csv but later might use meta learning so some text labels we
            # don't want to train/evaluate on...
            for set_type in ["_train.csv", "_dev.csv"]:
                file_path = os.path.join(self.data_dir, self.data_name + set_type)
                _, _, labels = self._read_csv_or_df(file_path)
                all_labels += flatten_list(labels)
            # Return the sorted unique values (as strings) of all the labels
            # Note that list(set(all_labels)) returns the unique labels as a list
            return sorted(list(map(str, list(set(all_labels)))))

    def _create_examples(self, texts_a, texts_b, labels, set_type):
        """
        Creates examples for the training and dev sets.

        Parameters
        ----------
        texts_a : list
            list of input texts_a (e.g the first `sentence` for BERT)
        texts_b : list, (list of None if not required)
            list of input texts_b (e.g the second `sentence` for BERT)
        labels : list
            list of input labels
        set_type : str
            specifies whether the set is 'train' or 'dev'

        Returns
        -------
        list
            list of InputExample objects

        Raises
        ------
        ValueError
            if the length of the texts and length of the labels are
            incompatible
        """
        examples = []
        for i, (text_a, text_b, label) in enumerate(zip(texts_a, texts_b, labels)):
            unique_id = f"{self.data_name}-{set_type}-{i}"
            text_a = convert_to_unicode(text_a)
            if text_b is not None:
                text_b = convert_to_unicode(text_b)
            if label is not None:
                if isinstance(label, str) and '[' and ']' in label:
                    # If we have a list of labels as a string convert into an actual list e.g
                    # "['list', 'of', 'strings']"" -> ['list', 'of', 'strings']
                    label = ast.literal_eval(label)
                label = convert_to_unicode(label)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a,
                             text_b=text_b, label=label))
        return examples

    def _read_csv_or_df(self, file_path_or_df):
        """
        Reads our standard CSV format, which we call a preprocessed_df (dataframe):
        The preprocessed dataframe will contain the (unique) `id` as an index,
        `text_a` as a column (first `sentence` in BERT),
        `text_b` as an OPTIONAL column (second `sentence`, if applicable) and `label` as a column
        The id's arent returned, not that relevant since we generate a fully unique id
        in _create_examples, but could be changed if required

        Parameters
        ----------
        file_path_or_df : str
            path to input CSV file or pandas dataframe

        Returns
        -------
        tuple
            first element is the texts_a in a list i.e ['text_a_1','text_a_2',...,'text_a_N']
            second element is the texts_b in a list i.e ['text_b_1','text_b_2',...,'text_b_N']
            (or [None,None,...,None] if there are no texts_b)
            corresponding labels of each text as a list i.e ['label_1','label_2',...,'label_N']
        """
        def process_labels_list(labels_list):
            """
            Processes the labels list, since if we have a list of strings of lists we want to
            convert the strings of lists into actual lists, else if just a regular list of labels
            it should pass through unchanged

            Parameters
            ----------
            labels_list : list
                list of the input labels
            """
            processed_labels_list = []
            for label in labels_list:
                if isinstance(label, str) and '[' and ']' in label:
                    # If we have a list of labels as a string convert into an actual list e.g
                    # "['list', 'of', 'strings']"" -> ['list', 'of', 'strings']
                    label = ast.literal_eval(label)
                processed_labels_list.append(label)
            return processed_labels_list

        if isinstance(file_path_or_df, str):
            # Open the file path
            with open(file_path_or_df, "r") as data_file:
                df = pd.read_csv(data_file, index_col=0)
        else:
            df = file_path_or_df
        if self.labelled_data:
            if len(df.columns) == 2:  # i.e it just contains the `text_a` and the `label`
                return df['text_a'].tolist(), [None] * len(df), process_labels_list(df['label'].tolist())
            elif len(df.columns) == 3:  #i.e it also includes `text_b`
                return df['text_a'].tolist(), df['text_b'].tolist(), process_labels_list(df['label'].tolist())
        else:
            if len(df.columns) == 1:  # i.e it just contains the `text_a` and NO label
                return df['text_a'].tolist(), [None] * len(df), [None] * len(df)
            elif len(df.columns) == 2:  #i.e it also includes `text_b` but still NO label
                return df['text_a'].tolist(), df['text_b'].tolist(), [None] * len(df)



class SST2Processor(DataProcessor):
    """
    Processor for the SST-2 data set.
    https://nlp.stanford.edu/sentiment/index.html
    """

    def __init__(self, data_dir):
        super().__init__(data_dir, "SST-2")


class SST5Processor(DataProcessor):
    """
    Processor for the SST-5 data set.
    https://nlp.stanford.edu/sentiment/index.html
    """

    def __init__(self, data_dir):
        super().__init__(data_dir, "SST-5")


class IMDBProcessor(DataProcessor):
    """
    Processor for the IMDB data set.
    https://ai.stanford.edu/~amaas/data/sentiment/
    """

    def __init__(self, data_dir):
        super().__init__(data_dir, "IMDB")

class SemEval_QA_MProcessor(DataProcessor):
    """
    Processor for the SemEval data set. Adapted from
    https://github.com/HSLCY/ABSA-BERT-pair
    """

    def __init__(self, data_dir):
        super().__init__(data_dir, 'SemEval_QA_M', 'Primary')

class SemEval_QA_BProcessor(DataProcessor):
    """
    Processor for the SemEval data set. Adapted from
    https://github.com/HSLCY/ABSA-BERT-pair
    """

    def __init__(self, data_dir):
        super().__init__(data_dir, 'SemEval_QA_B')

class Sentihood_QA_BProcessor(DataProcessor):
    """
    Processor for the Sentihood data set. Adapted from
    https://github.com/HSLCY/ABSA-BERT-pair
    """

    def __init__(self, data_dir):
        super().__init__(data_dir, 'Sentihood_QA_B')

class StreetbeesMoodProcessor(DataProcessor):
    """
    Processor for the Streetbees Mood Data. By default binary classification for each
    emotion category (0 = False, 1 = True)
    """

    def __init__(self, data_dir, labelled_data=True, labels=None, aspect_categories=None):
        super().__init__(data_dir, 'Streetbees_Mood', labelled_data=labelled_data, labels=labels)
        self.aspect_categories = ['Mindful', 'Motivated', 'Neutral', 'Anxious / worried',
                                  'Other', 'Energetic', 'Concerned', 'Cool', 'Healthy',
                                  'Positive', 'Relaxed', 'Busy', 'Thankful', 'Craving',
                                  'Cold', 'Sleepy / Tired', 'Hot', 'Full', 'Thirsty',
                                  'Rested / Awake', 'Good', 'Peaceful', 'Refreshed', 'Lazy',
                                  'Bored', 'Annoyed / angry', 'Sad / depressed', 'Thoughtful',
                                  'Great / amazing', 'Hungry', 'Unwell', 'Excited',
                                  'Happy / content', 'Fine'] if aspect_categories is None else aspect_categories

class NERProcessor(DataProcessor):
    """
    Processor for the CoNLL 2003 NER Task
    """

    def __init__(self, data_dir):
        super().__init__(data_dir, 'NER')
#                         labels=["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])

PROCESSORS = {'SST-2': SST2Processor, 'SST-5': SST5Processor,
              'IMDB': IMDBProcessor, 'SemEval_QA_B': SemEval_QA_BProcessor,
              'SemEval_QA_M': SemEval_QA_MProcessor, 'Streetbees_Mood': StreetbeesMoodProcessor,
              'NER': NERProcessor, 'Sentihood_QA_B': Sentihood_QA_BProcessor}
