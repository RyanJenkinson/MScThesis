import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset, TensorDataset
import os
import torch
from extract_features import convert_examples_to_features
from modelling import TOKENIZERS
from data_processing import DataProcessor
import pdb
class MetaLearningDataset(Dataset):
    def __init__(self, split, args):
        """
        Initialise the MetaLearningDataset class

        Parameters
        ----------
        split : str
            'train' or 'test'
        args : parsed args
            args passed in to the model
        """
        self.data_dir = "../../../datascience-projects/internal/multitask_learning/processed_data/Streetbees_Mood/Streetbees_Mood_all.csv"
        data = pd.read_csv(self.data_dir, index_col=0)
        self.data_processor = DataProcessor(data_dir=None, data_name='Streetbees_Mood', labels=['0','1'])
        self.K = args.K
        self.num_classes = args.num_classes

        self.baseLM_name = args.model_name.split("-")[0]
        do_lower_case = False if args.model_name.split("-")[-1] == "cased" else True
        self.tokenizer = TOKENIZERS[self.baseLM_name].from_pretrained(args.model_name, do_lower_case=do_lower_case)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pick categories for dev and test set
        ALL_CATEGORIES = np.unique(data['category'])
        TEST_CATEGORIES = ['Healthy', 'Positive', 'Unwell', 'Fine']

        self.tasks = {}
        for category in ALL_CATEGORIES:
            if (split == 'train' and category in TEST_CATEGORIES) or (split == 'test' and category not in TEST_CATEGORIES):
                continue
            # Each category will become a separate task. Explicitly redefine variable below for clarity
            task = category
            pos_examples, neg_examples = self.get_positive_and_negative_examples(data, category=task)
            if task not in self.tasks:
                self.tasks[task] = (pos_examples, neg_examples)

        task_list = []
        task_names = []
        for task in self.tasks:
            pos_examples = self.tasks[task][0]
            neg_examples = self.tasks[task][1]
            if len(pos_examples) < self.K or len(neg_examples) < self.K:
                print('not enough examples', task)
                continue # skip for now if not enough examples
            task_list.append((pos_examples, neg_examples))
            task_names.append(task)

        self.tasks = task_list
        self.task_names = task_names
        self.num_tasks = len(self.tasks)

    @staticmethod
    def get_positive_and_negative_examples(data, category):
        positive_examples = data[(data['category'] == category) & (data['label'] == 1)]
        negative_examples = data[(data['category'] == category) & (data['label'] == 0)]
        return positive_examples.drop(columns='category'), negative_examples.drop(columns='category')

    def __getitem__(self, task_index):
        # choose the task indicated by index
        pos_examples, neg_examples = self.tasks[task_index]

        # for now just choose randomly among examples
        pos_indices = np.random.choice(range(len(pos_examples)), size=self.K)
        neg_indices = np.random.choice(range(len(neg_examples)), size=self.K)

        # # interleave randomly - DIFFERENT FROM RANDOMLY SHUFFLING
        # examples = np.empty((self.K*2, 2), dtype=pos.dtype)
        # if np.random.uniform() > .5:
        #     examples[0::2, :] = pos
        #     examples[1::2, :] = neg
        # else:
        #     examples[0::2, :] = neg
        #     examples[1::2, :] = pos

        # Randomly shuffle positive and negative examples for now
        all_examples = pd.concat([pos_examples.iloc[pos_indices, :],
                                  neg_examples.iloc[neg_indices, :]]).sample(frac=1)
        train_examples = self.data_processor.get_examples(input_df=all_examples.iloc[:self.K, :], set_type='train')
        test_examples = self.data_processor.get_examples(input_df=all_examples.iloc[self.K:, :], set_type='test')

        train_features = convert_examples_to_features(train_examples, label_list=['0','1'], max_seq_length=128,
                                                      tokenizer=self.tokenizer, task_name='Streetbees_Mood',
                                                      model_name=self.baseLM_name, do_logging=False)
        all_input_ids = torch.tensor([feature.input_ids for feature in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([feature.segment_ids for feature in train_features], dtype=torch.long)
        all_input_masks = torch.tensor([feature.input_mask for feature in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([feature.label_id for feature in train_features], dtype=torch.long)
        task_train_data = {'input_ids': all_input_ids, 'segment_ids': all_segment_ids,
                           'input_masks':  all_input_masks, 'label_ids': all_label_ids}

        test_features = convert_examples_to_features(test_examples, label_list=['0','1'], max_seq_length=128,
                                                      tokenizer=self.tokenizer, task_name='Streetbees_Mood',
                                                      model_name=self.baseLM_name, do_logging=False)
        all_input_ids = torch.tensor([feature.input_ids for feature in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([feature.segment_ids for feature in train_features], dtype=torch.long)
        all_input_masks = torch.tensor([feature.input_mask for feature in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([feature.label_id for feature in train_features], dtype=torch.long)
        task_test_data = {'input_ids': all_input_ids, 'segment_ids': all_segment_ids,
                          'input_masks':  all_input_masks, 'label_ids': all_label_ids}

        return task_train_data, task_test_data

    def __len__(self):
        return len(self.tasks)
