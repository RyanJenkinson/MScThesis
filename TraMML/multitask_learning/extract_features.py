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
PyTorch BERT model. The following code is inspired by the HuggingFace version
and it is built upon in places (license below). In particular it can be
used and modified for commercial use:
https://github.com/huggingface/pytorch-pretrained-BERT/
In this code we can extract the feature vectors from the BERT encodings.
"""

import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)

FEATURE_SETTINGS = {'bert': {'cls_token_at_end': False,
                             'pad_on_left': False,
                             'cls_token': '[CLS]',
                             'sep_token': '[SEP]',
                             'pad_token': 0,
                             'sequence_a_segment_id': 0,
                             'sequence_b_segment_id': 1,
                             'cls_token_segment_id': 0,
                             'pad_token_segment_id': 0,
                             'mask_padding_with_zero': True},
                    'xlnet': {'cls_token_at_end': True,
                              'pad_on_left': True,
                              'cls_token': '<cls>',
                              'sep_token': '<sep>',
                              'pad_token': 0,
                              'sequence_a_segment_id': 0,
                              'sequence_b_segment_id': 1,
                              'cls_token_segment_id': 2,
                              'pad_token_segment_id': 4,
                              'mask_padding_with_zero': True}
                    }
class InputExample(object):
    """A class to store a single input example"""
    def __init__(self, unique_id, text_a, text_b=None, label=None):
        """A wrapper for an input example

        Parameters
        ----------
        object : [type]
            [description]
        unique_id : int
            The unique id for the example
        text_a : str
            The first sentence
        text_b : int, optional
            The second sentence, by default None
        label : str, optional
            label for the example, required for training and dev sets.
            Not required for the test set
        """
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """The convention in BERT is:
    a) For sequence pairs:
    tokens:      [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    segment_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1  1
    b) For single sequences:
    tokens:       [CLS] the dog is hairy . [SEP]
    segment_ids:   0     0   0   0   0   0  0

    Where "segment_ids" are used to indicate whether this is the first
    sequence or the second sequence. The embedding vectors for `type=0` and
    `type=1` were learned during pre-training and are added to the wordpiece
    embedding vector (and position vector). This is not *strictly* necessary
    since the [SEP] token unambigiously separates the sequences, but it makes
    it easier for the model to learn the concept of sequences.

    label_id is the id of the corresponding label for the sentence classification/regression problem

    For NER Tasks, we define label_id for each word ay the positions of the first part of the wordpiece for classification,
    label_id is now a list, with -1 corresponding to no label required
    tokens:      [CLS] is this jack ##son ##ville ? [SEP]
    segment_ids:   0   0   0    0    0     0      0   0
    label_id:    [-1,['O'],['O'],['B-LOC'],-1,-1,['O'],-1]
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, task_name=None, model_name=None):
    """
    Loads a data file into a list of `InputBatch`s.
    Some potential speedups have been noted and commented but this code tried to remain
    faithful to the huggingface implementation


    Parameters
    ----------
    examples : list
        List of InputExample objects
    label_list : list
        List of labels typically returned from the data processors
        NOTE: If task_name == 'NER' then label_list is a list of labels corresponding to a label for each
        word in the text_a sequence
    max_seq_length : int
        Maximum sequence length (to zero pad up to)
    tokenizer : object
        Tokenizer object, typically BERTTokenizer with do_lower_case = True/False
        as appropriate
    task_name : str
        Task name, since for NER we convert examples to different features, by default None
    model_name : str
        The name of the model we are using (BERT of XLNet) which will inform our tokenization procedure

    Returns
    -------
    list
        list of InputFeatures compatible with BERT or XLNet
    """

    # Create unique dictionary of labels and their corresponding map
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # Get local variables from the dictionary of feature settings
    (cls_token_at_end, pad_on_left, cls_token, sep_token,
     pad_token, sequence_a_segment_id, sequence_b_segment_id,
     cls_token_segment_id, pad_token_segment_id, mask_padding_with_zero) = FEATURE_SETTINGS[model_name].values()

    features = []
    for (ex_index, example) in enumerate(examples):
        if task_name == 'NER':
            text_list = example.text_a.split(' ')
            tokens_a = []
            label_id = []
            for word_idx, word in enumerate(text_list):
                # Get the wordpiece tokens (possibly more than 1) and extend the tokens_all list
                word_tokens = tokenizer.tokenize(word)
                tokens_a.extend(word_tokens)
                # Mark the start of the wordpieces as valid, and get the corresponding word label; all other word pieces thereafter are invalid
                label_id.extend([label_map[example.label[word_idx]]] + [-1]*(len(word_tokens) - 1))
        else:
            tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                if task_name == 'NER':
                    label_id = label_id[:(max_seq_length - 2)]

        # Set up the tokens and put them through the tokenizer
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Get the corresponding label id for the example, or a list of ids if task_name == 'NER'
        if task_name == 'NER':
            # Account for the "CLS" and "SEP" tokens
            if cls_token_at_end:
                label_id = label_id + [-1] + [-1]
            else:
                label_id = [-1] + label_id + [-1]
        else:
            label_id = label_map[example.label] if example.label is not None else None

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if task_name == 'NER':
            label_padding_length = max_seq_length - len(label_id)

        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            if task_name == 'NER':
                label_id = [-1]*label_padding_length + label_id
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            if task_name == 'NER':
                label_id = label_id + [-1]*label_padding_length

        # Make sure everything has the right shape
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if task_name == 'NER':
            assert len(label_id) == max_seq_length

        # log some examples to make sure everything is working
        if ex_index < 3:
            # We use Python 3.6 f-strings for nice formatting!
            LOGGER.info("*** Example ***")
            LOGGER.info(f"unique_id: {example.unique_id}")
            LOGGER.info(f"tokens: {' '.join(str(token) for token in tokens)}")
            LOGGER.info(f"input_ids: {' '.join(str(input_id) for input_id in input_ids)}")
            LOGGER.info(f"input_mask: {' '.join(str(mask) for mask in input_mask)}")
            LOGGER.info(f"segment_ids: {' '.join(str(segment_id) for segment_id in segment_ids)}")
            if task_name == 'NER':
                LOGGER.info(f"labels: {' '.join(str(label) for label in example.label)} "
                            f"(ids: {' '.join(str(lab_id) for lab_id in label_id)})")
            else:
                LOGGER.info(f"label: {example.label} (id {label_id})")

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """A simple function to truncate a sequence pair in place to a max length
    We truncate the longer of the two sentences one token at a time in an
    attempt to maintain the most amount of information possible
    Parameters
    ----------
    tokens_a : array
        First sentence as a sequence of tokens
    tokens_b : array
        Second sentence as a sequence of tokens
    max_length : int
        Maximum length of the combined sentence lengths
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
