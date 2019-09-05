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
This code will store all the modelling required to build up various BERT models
"""
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_transformers import BertModel, XLNetModel
from pytorch_transformers import BERT_PRETRAINED_MODEL_ARCHIVE_MAP, XLNET_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers import BertTokenizer, XLNetTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.modeling_utils import SequenceSummary
# Setup pretrained models to download - typically use bert-base or xlnet-base
MODEL_NAMES = list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys()) + list(XLNET_PRETRAINED_MODEL_ARCHIVE_MAP.keys())

MODELS = {'bert': BertModel, 'xlnet': XLNetModel}
TOKENIZERS = {'bert': BertTokenizer, 'xlnet': XLNetTokenizer}

class MultiTaskModel(nn.Module):
    """
    Custom class for Multitask objective using BERT as the base model.
    It automatically generates a set of heads, with a simple linear layer onto
    each task to give a classification according to the task config i.e
    task_configs[task_name]['num_labels']
    task_configs must be a dict of the following form:
    {task_name: {'num_labels': <int>, 'task_type': <str>, 'output_type', <str>} for task_name in tasks} where:
    num_labels : number of labels (int)
    task_type : indicates if a task is a primary, secondary or tertiary task. Not used in this class.
    output_type : indicates if task if regression or classification ('REG'/'CLS')
    """
    def __init__(self, task_configs, model_name_or_config='bert-base-cased'):
        super(MultiTaskModel, self).__init__()

        self.task_configs = task_configs

        # Load BERT model from either a specified config file or pretrained
        # Apply weights initialisation (automatic if from_pretrained)
        if isinstance(model_name_or_config, dict):
            # TODO: For now, assume if a config is passed in then it is applied to
            # the BERT model, may change in future
            self.baseLM = BertModel(config=model_name_or_config)
            self.baseLM.apply(self.__init_weights)
        elif isinstance(model_name_or_config, str):
            # Extract model name from the string
            model_name_or_config = model_name_or_config.lower()
            self.base_model_name = model_name_or_config.split("-")[0]
            if model_name_or_config not in MODEL_NAMES:
                raise ValueError(f"Please enter a valid model name - you entered {model_name_or_config}")
            self.baseLM = MODELS[self.base_model_name].from_pretrained(model_name_or_config)

        # Store our config file
        self.baseLM_config = self.baseLM.config

        # Add the sequence summary module to compute a single vector summary of a sequence hidden states according to:
        #     summary_type:
        #         - 'last' => [default] take the last token hidden state (like XLNet)
        #         - 'first' => take the first token hidden state (like Bert)
        #         - 'mean' => take the mean of all tokens hidden states (NOT Masked Mean)
        #         - 'masked_mean' => take the masked mean (masked by the attention weights)
        self.summary_type = 'first' if self.base_model_name == 'bert' else 'last'
        self.activation = nn.Tanh()

        # Add the heads (just simple linear layers mapping to num_labels)
        self.heads = nn.ModuleDict({task_name:
                                    nn.Linear(self.baseLM_config.hidden_size,
                                              task_config['num_labels'])
                                    for task_name, task_config
                                    in self.task_configs.items()})

        # Apply initialisation to the heads of the model
        self.heads.apply(self.__init_weights)

        # Get the device the that we are using
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init_weights(self, module):
        """
        Initialise the weights of the PyTorch nn module according to
        bert_config

        Parameters
        ----------
        module : PyTorch nn.Module
            Module inside the neural net
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            # Perform the same initialisation protocol as that in bert_config
            module.weight.data.normal_(mean=0.0,
                                       std=self.baseLM_config.initializer_range)
        if (isinstance(module, (nn.Linear, nn.LayerNorm)) and
                module.bias is not None):
            module.bias.data.zero_()

    def prepare_optimizer_and_scheduler(self, num_train_optimization_steps,
                                        learning_rate=1e-6, warmup_proportion=0.1,
                                        weight_decay=0.0, adam_epsilon=1e-8):
        """
        Prepares Adam optimizer for our Langauge Model with the new behaviour in pytorch-transformers
        We have an optimizer and a scheduler

        Parameters
        ----------
        epochs : int
            number of training epochs
        num_train_optimization_steps : int
            number of training optimization steps
        learning_rate : float
            the learning rate for the Adam optimiser
        warmup_proportion : float
            proportion of the learning rate that is a ramp,
            i.e. learning_rate_fn(0.1*total_steps) -> 10% of training steps
            = max(learning_rate) = learning_rate input param
        weight_decay : float
            weight decay (L2) for all params except the biases and
            normalisation layers, default 0.0
        adam_epsilon : float
            Epsilon for Adam Optimizer, default 1e-8

        Returns
        ------
        optimizer : AdamW
            Adam Optimizer
        scheduler : WarmupLinearSchedule
            Warmup Schedule
        """
        # Prepare optimizer and scheduler (new behaviour in pytorch-transfomers)
        param_list = list(self.baseLM.named_parameters()) + list(self.heads.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_list
                        if not any(no_decay_name in name for no_decay_name in no_decay)],
             'weight_decay': weight_decay},
            {'params': [param for name, param in param_list
                        if any(no_decay_name in name for no_decay_name in no_decay)],
             'weight_decay': 0.0}
        ]
        # To reproduce BertAdam specific behavior set correct_bias=False
        # this avoids correcting bias in Adam (e.g. like in Bert TF repository)
        correct_bias = False if self.base_model_name == 'bert' else True
        num_warmup_steps = num_train_optimization_steps * warmup_proportion
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                          correct_bias=correct_bias, eps=adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps,
                                         t_total=num_train_optimization_steps)
        return optimizer, scheduler

    def unfreeze_base_layers(self, base_params_to_unfreeze='all'):
        """
        Function to automatically freeze certain layers of the base model

        Parameters
        ----------
        base_params_to_unfreeze : list or str, optional
            List or str of parameters to unfreeze. It will match any part of the
            parameter name automatically outputted by pytorch. To see the list
            of module names run the following code:
            ```
            model = MultiTaskModel()
            param_names = [name for name, _ in model.named_parameters()]
            ```
            Note that the parameter names are separated/indexed by full stops
            by default 'all' (i.e every param is unfrozen)
            If you wanted a specific layer then 'layer.11' could be passed, for example
        """
        for name, param in self.baseLM.named_parameters():
            if ('all' in base_params_to_unfreeze or
                    any(unfreeze_param in name for unfreeze_param in base_params_to_unfreeze)):
                param.requires_grad = True
            else:
                param.detach()
                param.requires_grad = False

    def forward(self, input_ids, segment_ids, attention_mask, task_name, labels=None):
        """
        The forward pass for the MultiTaskModel module

        Parameters
        ----------
        input_ids : tensor
            PyTorch tensor of IDs
        segment_ids : tensor
            PyTorch tensor of segment IDs - 0 = text_a, 1 = text_b
        attention_mask : tensor
            PyTorch tensor of the attention mask -  Mask of 1's over input text
        task_name : str
            Name of the task to forward pass through
        labels : tensor, optional
            PyTorch tensor of labels for the relevant task, by default None

        Returns
        -------
        tensor or tuple of tensors
            If labels is not None: output is loss, logits where the appropriate
            loss function (for classification or regression) has been chosen
            automatically based on the task config.
            If labels is None: return the logits
        """
        # Get outputs from the model
        model_outputs = self.baseLM(input_ids, segment_ids, attention_mask=attention_mask)
        # Get the last_hidden_state: ``torch.FloatTensor`` shape ``(batch_size, sequence_length, hidden_size)``
        sequence_output = model_outputs[0]
        if task_name == 'NER':
            pooled_output = sequence_output
        else:
            if self.summary_type == 'last':
                pooled_output = sequence_output[:, -1]
            elif self.summary_type == 'first':
                pooled_output = sequence_output[:, 0]
            elif self.summary_type == 'mean':
                pooled_output = sequence_output.mean(dim=1)
            elif self.summary_type == 'masked_mean':
                # Alternative is to compute a simple (masked) average to pool the output, sum along sequence_length axis and then divide by number of active tokens
                pooled_output = torch.einsum('ijk, ij -> ik', sequence_output, attention_mask.float())
                pooled_output /= torch.sum(attention_mask.float(), dim=1)[:, None]
            pooled_output = self.activation(pooled_output)
        # Forward pass through the head of the corresponding task
        logits = self.heads[task_name](pooled_output)
        # TODO: New addition to the code online is reshaped logits, does this matter?

        if labels is not None:
            if self.task_configs[task_name]["output_type"] == "CLS":
                # We have to ignore the 0's in the valid_output so that they dont contribute to the gradient
                # This is only relevant for the NER task, else we set it to the default for PyTorch, which is -100
                ignore_index = -1 if task_name == 'NER' else -100
                loss_fn = CrossEntropyLoss(ignore_index=ignore_index)
                loss = loss_fn(logits.view(-1, self.task_configs[task_name]['num_labels']), labels.view(-1))
                return loss, logits
            elif self.task_configs[task_name]["output_type"] == "REG":
                loss_fn = MSELoss()
                loss = loss_fn(logits, labels.unsqueeze(1))
                return loss, logits
        else:
            return logits
