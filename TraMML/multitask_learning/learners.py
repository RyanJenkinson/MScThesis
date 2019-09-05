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
This code will run the BERT multitask learning training and evaluation scripts
based on the config file
"""
import json
import logging
import random
from pathlib import Path
from itertools import cycle
from tqdm import tqdm, trange

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tensorboardX import SummaryWriter

from data_processing import PROCESSORS
from extract_features import convert_examples_to_features
from modelling import MultiTaskModel, MODEL_NAMES, TOKENIZERS
from evaluation import Macro_PRF, SemEval_acc, Sentihood_strict_acc, Sentihood_AUC_Acc, NER_eval, TASK_EVALUATION_SETTINGS

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)

TASK_WEIGHTINGS = ['num_examples_per_task', 'importance']
SAMPLING_MODES = ['sequential', 'random', 'prop', 'sqrt', 'square', 'anneal']
# TODO: Other sampling modes ? e.g reverse anneal

class MultiTaskLearner:
    """
    Wrapper for running Multi Task Learning built on top of HuggingFace's PyTorch implementation
    """
    def __init__(self, config):
        """
        Initialise the model with the config

        Parameters
        ----------
        config : dict
            Settings for the training run/loop (see below for parameters)
        """
        # Set up the run config
        config["data_dir"] = config.get("data_dir", None)
        if config["data_dir"] is None:
            raise ValueError("""Please enter {'data_dir': 'path/to/data'} in
                             the run config JSON file""")
        config["log_dir"] = config.get("log_dir", None)

        config["tasks"] = config.get("tasks", "SST-2")
        config["model_name"] = config.get("model_name", None)
        config["model_config"] = config.get("model_config", None)
        if config["model_name"] is None:
            if config['model_config'] is None:
                raise ValueError("You must enter one of (model_name, model_config) to the run config")
            if not isinstance(config["model_config"], dict):
                raise TypeError(f"The model config must be a dict! You entered type: {type(config['model_config'])}")
        else:
            config["model_name"] = config["model_name"].lower()
        config["base_params_to_unfreeze"] = config.get("base_params_to_unfreeze", "all")
        # TODO: Can we infer max_seq_length from the data?
        config["max_seq_length"] = config.get("max_seq_length", 128)
        config["train_batch_size"] = config.get("train_batch_size", 24)
        config["dev_batch_size"] = config.get("dev_batch_size", 128)
        config["learning_rate"] = config.get("learning_rate", 2e-5)
        config["warmup_prop"] = config.get("warmup_prop", 0.1)
        config["num_epochs"] = config.get("num_epochs", 4)
        config["sampling_mode"] = config.get("sampling_mode", "sequential")
        if config["sampling_mode"] == "anneal":
            config["anneal_constant"] = config.get("anneal_constant", 0.9)
        config["task_weightings"] = config.get("task_weightings", "importance")
        config["steps_to_log"] = config.get("steps_to_log", 500)
        config["seed"] = config.get("seed", 42)

        self.config = config

        # Check the base model is a valid name
        if self.config["model_name"] not in MODEL_NAMES:
            raise ValueError(f"Please enter a valid model name - you entered {self.config['model_name']} "
                             f"- try one of {MODEL_NAMES}")
        # Get the device we are working on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()

        # TODO: Implement distributed/multiple GPU support if running in the cloud/open sourcing?

        # Set the random seed for somewhat reproducible results
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if n_gpu > 0:
            torch.cuda.manual_seed_all(config["seed"])

        # Get some useful values and log our task setup according to config
        self.task_names = list(self.config["tasks"].split(", "))
        self.num_tasks = len(self.task_names)

        if any(task_name not in PROCESSORS.keys() for task_name in self.task_names):
            raise ValueError(f"Non implemented task string in config please try one of {PROCESSORS.keys()}")

        LOGGER.info(f"Number of tasks: {self.num_tasks}, "
                    f"Name(s): {', '.join(self.task_names)}")

        # Set up variables for loading in the data via load_data. The split takes into account Sentihood and SemEval tasks
        self.data_dirs = [Path(self.config["data_dir"]) / task_name.split("_")[0] for task_name in self.task_names]
        self.train_loaders = []
        self.dev_loaders = []
        self.train_examples_per_task = [None] * self.num_tasks  # Updated in load_data()
        self.data_loaded = False
        self.processor_list = [PROCESSORS[task_name](data_dir)
                               for task_name, data_dir in zip(self.task_names, self.data_dirs)]
        self.label_list = {processor.data_name: processor.get_labels() for processor in self.processor_list}

        # Get our specific task configs based on the tasks we want to run
        self.task_configs = {task_name: {"num_labels": processor.num_labels,
                                         "task_type": processor.task_type,
                                         "output_type": processor.output_type}
                             for task_name, processor in zip(self.task_names, self.processor_list)}

        # Create the writer, logging key hyperparameters in the log name and all parameters as text
        hparams_in_logname = {'LR': self.config["learning_rate"], 'SM': self.config["sampling_mode"],
                              'BS': self.config["train_batch_size"], 'Tasks': "|".join(self.task_names)}
        logname = '_'.join('{}_{}'.format(*param) for param in sorted(hparams_in_logname.items()))
        self.writer = SummaryWriter(comment=logname, log_dir=self.config["log_dir"])
        for config_param, config_value in self.config.items():
            self.writer.add_text(str(config_param), str(config_value))



        # Instantiate the model and save it to the device (CPU or GPU)
        model_name_or_config = (self.config["model_config"] if self.config["model_config"] is not None
                                 else self.config["model_name"])
        self.model = MultiTaskModel(task_configs=self.task_configs, model_name_or_config=model_name_or_config)
        #self.model.unfreeze_base_layers(base_params_to_unfreeze=self.config["base_params_to_unfreeze"])
        self.model.to(self.device)
        self.baseLM_name = self.model.base_model_name

        # Initialise the tokenizer based on the model fed in
        do_lower_case = False if self.config["model_name"].split("-")[-1] == "cased" else True
        self.tokenizer = TOKENIZERS[self.baseLM_name].from_pretrained(config["model_name"], do_lower_case=do_lower_case)

        model_params = sum(param.numel() for param in self.model.parameters())
        trained_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        LOGGER.info(f"Model initialised with {model_params:3e} parameters , of which {trained_params:3e} "
                    f"are trainable i.e. {100 * trained_params/model_params:.3f}%")

        # Task weighting mode verification
        if self.config["task_weightings"] not in TASK_WEIGHTINGS:
            raise ValueError(f"Non implemented task weighting mode in config, please try one of {TASK_WEIGHTINGS}")
        self.task_weightings_mode = self.config["task_weightings"]

        # Sampling mode verification
        if self.config["sampling_mode"] not in SAMPLING_MODES:
            raise ValueError(f"Non implemented sampling mode in config, please try one of {SAMPLING_MODES}")
        self.sampling_mode = self.config["sampling_mode"]

    def load_data(self):
        """
        Loads training and test data for the tasks given in config
        into PyTorch DataLoader iterators
        """
        # Load the training and dev examples for each task
        train_examples = {processor.data_name: processor.get_examples(set_type="train") for processor in self.processor_list}
        self.train_examples_per_task = [len(train_examples[task]) for task in self.task_names]
        dev_examples = {processor.data_name: processor.get_examples(set_type="dev") for processor in self.processor_list}

        train_bs = self.config["train_batch_size"]
        dev_bs = self.config["dev_batch_size"]
        max_seq_length = self.config["max_seq_length"]
        # TODO: Add in DistributedSampler if using more than 1 gpu and torch.nn.parallel.DistributedDataParallel
        for task_name in self.task_names:
            label_dtype = torch.long if self.task_configs[task_name]['output_type'] == 'CLS' else torch.float
            train_features = convert_examples_to_features(train_examples[task_name], self.label_list[task_name],
                                                          max_seq_length, self.tokenizer, task_name, self.baseLM_name)
            all_input_ids = torch.tensor([feature.input_ids for feature in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([feature.segment_ids for feature in train_features], dtype=torch.long)
            all_input_masks = torch.tensor([feature.input_mask for feature in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([feature.label_id for feature in train_features], dtype=label_dtype)
            task_train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_masks, all_label_ids)

            train_sampler = RandomSampler(task_train_data)
            self.train_loaders.append(iter(DataLoader(task_train_data, sampler=train_sampler, batch_size=train_bs)))
        # Load in the dev set
        for task_name in self.task_names:
            label_dtype = torch.long if self.task_configs[task_name]['output_type'] == 'CLS' else torch.float
            dev_features = convert_examples_to_features(dev_examples[task_name], self.label_list[task_name],
                                                        max_seq_length, self.tokenizer, task_name, self.baseLM_name)
            all_input_ids = torch.tensor([feature.input_ids for feature in dev_features], dtype=torch.long)
            all_segment_ids = torch.tensor([feature.segment_ids for feature in dev_features], dtype=torch.long)
            all_input_masks = torch.tensor([feature.input_mask for feature in dev_features], dtype=torch.long)
            all_label_ids = torch.tensor([feature.label_id for feature in dev_features], dtype=label_dtype)
            task_dev_data = TensorDataset(all_input_ids, all_segment_ids, all_input_masks, all_label_ids)

            self.dev_loaders.append(DataLoader(task_dev_data, batch_size=dev_bs))

        self.data_loaded = True

    def train(self, save_model_every_epoch=True):
        """
        Trains the Multitask learning model with the following settings
        as defined in the run config:
        sampling_mode - The way in which we choose the next task in each step over
                        the epoch, sampling from the task_weightings distribution
        num_epochs - Number of epochs to train for
        steps_per_epoch - Number of optimizer steps to take per epoch

        Parameters
        ----------
        save_model_every_epoch : bool, optional
            Whether or not to save the model after every epoch, by default True

        Returns
        -------
        tuple
            final_loss_train, final_loss_dev, final_acc_dev
        """
        def get_task_ids(task_weightings, num_epochs, steps_per_epoch, sampling_mode='sequential'):
            """
            A helper function for getting task_ids sampled from the distribution defined via sampling_mode

            Parameters
            ----------
            task_weightings : list of floats
                The weightings of the tasks that we transform into a distribution via sampling mode from which
                we sample the task_ids
            num_epochs : int
                The number of training epochs
            steps_per_epoch : int
                The number of steps per training epoch
            sampling_mode : str, optional
                How to sample the task_ids, by default 'sequential'

            Returns
            ----------
            task_ids : list of lists/arrays
                A list indexed as task_ids[e][s] that tells you the task id for step s in epoch e
            """
            alphas = {'random': 0, 'prop': 1, 'sqrt': 0.5, 'square': 2}

            if sampling_mode == 'sequential':
                task_ids = [[step % self.num_tasks for step in range(steps_per_epoch)] for epoch in range(num_epochs)]

            if sampling_mode not in ['sequential', 'anneal']:
                alpha = alphas[sampling_mode]
                probs = [weight**alpha for weight in task_weightings]
                probs = [prob/sum(probs) for prob in probs]
                task_ids = np.random.choice(self.num_tasks, size=[num_epochs, steps_per_epoch], p=probs)
            elif sampling_mode == 'anneal':
                anneal_constant = self.config["anneal_constant"]
                # Generate the list by looping over the epochs, since the alpha depends on the epoch we are on
                task_ids = []
                for epoch in range(num_epochs):
                    alpha = (1 - anneal_constant*(epoch/num_epochs))
                    probs = [weight**alpha for weight in task_weightings]
                    probs = [prob/sum(probs) for prob in probs]
                    task_ids.append(np.random.choice(self.num_tasks, size=steps_per_epoch, p=probs))
            return task_ids

        # Load the data from .load_data() if not already loaded
        if not self.data_loaded:
            self.load_data()

        # Cycle train_loaders iterations ready for training
        self.train_loaders = [cycle(it) for it in self.train_loaders]

        if self.task_weightings_mode == "num_examples_per_task":
            task_weightings = self.train_examples_per_task
        else:
            task_types = [self.task_configs[task_name]["task_type"] for task_name in self.task_names]
            # TODO: Add in flexibility/functionality for the below map?
            task_type_map = {'Primary': 4, 'Secondary': 2, 'Tertiary': 1}
            task_weightings = [task_type_map[task_type] for task_type in task_types]

        num_epochs = self.config["num_epochs"]
        # TODO: Check most principled way for steps_per_epoch. Max, sum? Might need readjusting factor...
        # Readjusting factor might be proportional to the number of tasks i.e self.num_tasks
        # steps_per_epoch = int((max(self.train_examples_per_task) / self.config["train_batch_size"]))
        steps_per_epoch = int((np.mean(self.train_examples_per_task) * self.num_tasks / self.config["train_batch_size"]))
        num_training_steps = steps_per_epoch * num_epochs

        optimizer, scheduler = self.model.prepare_optimizer_and_scheduler(num_training_steps,
                                                                          learning_rate=self.config["learning_rate"],
                                                                          warmup_proportion=self.config["warmup_prop"])

        # Initialise the global step and generate task_ids according to our sampling mode, then start the training loop
        train_global_step = {task_name: 0 for task_name in self.task_names}
        eval_global_step = {task_name: 0 for task_name in self.task_names}
        task_ids = get_task_ids(task_weightings, num_epochs, steps_per_epoch, self.sampling_mode)
        # task_names = np.vectorize(lambda task_id: self.task_names[task_id])(task_ids)  # To map onto names instead of ids
        # task_order = {'Epoch ' + str(epoch): task_names[epoch] for epoch in range(num_epochs)}
        for epoch in trange(num_epochs, desc="Epoch"):
            # Make the model trainable again after evaluating every epoch
            self.model.train()
            self.model.zero_grad()

            # Initialise/reset train_losses, train_steps and num_examples per task
            train_losses = {task_name: 0 for task_name in self.task_names}
            n_train_steps, n_train_examples = 0, 0
            for step in trange(steps_per_epoch, desc="Step"):
                # Get the task_id from our generated list, as well as the task name, load the appropriate
                # batch from the right task, run it through the model and backprop the loss
                task_id = task_ids[epoch][step]
                task_name = self.task_names[task_id]
                batch = next(self.train_loaders[task_id])
                batch = tuple(model_input.to(self.device) for model_input in batch)
                input_ids, segment_ids, input_masks, label_ids = batch
                loss, _ = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)
                loss.backward()
                scheduler.step()  # Update learning rate scheduler
                optimizer.step()  # Update optimizer
                self.model.zero_grad()

                # Increment all relevant values (mean will aggregate if computation over multiple GPUs)
                train_losses[task_name] += loss.mean().item()
                n_train_examples += input_ids.size(0)
                n_train_steps += 1
                train_global_step[task_name] += 1
                if step % self.config["steps_to_log"] == 0:
                    LOGGER.info(f"Task: {task_name} - Step: {step} - Loss: {train_losses[task_name]/n_train_steps}")

                # Log training batch statistics to tensorboard
                self.writer.add_scalars('loss', {task_name + '_train_loss': train_losses[task_name]/n_train_steps},
                                        train_global_step[task_name])

            # After the epoch, normalise the accumulated training losses and set eval accs/losses to 0 for updating
            train_losses = {task_name: train_loss / n_train_steps for task_name, train_loss in train_losses.items()}
            eval_accs = {task_name: 0 for task_name in self.task_names}
            eval_losses = {task_name: 0 for task_name in self.task_names}
            metrics_reports = {task_name: None for task_name in self.task_names}

            # Run the evaluation method for each of the tasks
            for task_id, task_name in enumerate(self.task_names):
                eval_accs[task_name], eval_losses[task_name], n_steps, metrics_reports[task_name] = self.evaluate_model(task_id,
                                                                                       				        eval_global_step[task_name])
                eval_global_step[task_name] += n_steps

            # Log the results
            result_dict = {'train_steps': sum(train_global_step.values()), 'train_loss': train_losses,
                           'eval_losses': eval_losses, 'eval_accuracies': eval_accs}
            LOGGER.info(f"End of epoch {epoch+1} - Results: {result_dict}")

            if save_model_every_epoch:
                torch.save(self.model.state_dict(), 'saved_models/' + "_".join(self.task_names + [self.sampling_mode]) + '.pt')

        final_loss_train, final_loss_dev, final_acc_dev, final_metrics_reports = train_losses, eval_losses, eval_accs, metrics_reports
        return final_loss_train, final_loss_dev, final_acc_dev, final_metrics_reports

    def evaluate_model(self, task_id, eval_global_step, data_loaders=None, return_preds=False):
        """
        Evaluation logic for the Multitask learning model

        Parameters
        ----------
        task_id : int
            The id (starting from 0) of the corresponding task
        eval_global_step : int
            Global step for number of evaluations taken so far (on previous epochs)
        data_loaders : list of PyTorch DataLoader object, one for each task (optional)
            Evaluates the model with the data in the dataloader, default None, where it defaults
            to evaluating on the dev set (used in the training process to report scores)
        return_preds : bool (optional)
            Whether or not to return the predictions, default False

        Returns
        -------
        float
            Evaluation accuracy over the dev set
        """
        self.model.eval()
        task_name = self.task_names[task_id]
        split_task_name = task_name.split("_")[0]
        if data_loaders is None:
            data_loaders = self.dev_loaders

        # Evaluation for SemEval/Sentihood (TABSA) is more complicated, so will require additional evaluation logic
        if split_task_name in ['SemEval', 'Sentihood']:
            eval_loss, n_eval_steps = 0, 0
            # Get the evaluation labels
            y_pred = []
            y_true = []
            scores = None
            for batch in tqdm(data_loaders[task_id], "Evaluating"):
                input_ids, segment_ids, input_masks, label_ids = tuple(model_input.to(self.device)
                                                                       for model_input in batch)
                with torch.no_grad():
                    loss, logits = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)

                eval_loss += loss.mean().item()
                logits = logits.detach().cpu().numpy()
                y_pred_batch = np.argmax(logits, axis=1)
                y_true_batch = label_ids.to('cpu').numpy()

                y_pred = np.concatenate([y_pred, y_pred_batch])
                y_true = np.concatenate([y_true, y_true_batch])
                if scores is None:
                    scores = logits
                else:
                    scores = np.concatenate([scores, logits])

                n_eval_steps += 1

                # Log evaluation loss to tensorboard
                self.writer.add_scalars('loss', {task_name + '_eval_loss': eval_loss/n_eval_steps},
                                        eval_global_step + n_eval_steps)
            global_acc = accuracy_score(y_true, y_pred)
            if task_name[-1] == 'B':  # If we are in the binary setting then perform post processing
                scores = scores[:,1].reshape(-1,TASK_EVALUATION_SETTINGS[split_task_name]['num_sent_categories'])
                y_pred = np.argmax(scores, axis=1)
                y_true = np.argmax(y_true.reshape(-1,TASK_EVALUATION_SETTINGS[split_task_name]['num_sent_categories']),
                                   axis=1)
            aspect_acc = accuracy_score(y_true, y_pred)

            if split_task_name == 'SemEval':
                if task_name[-1] == 'B':
                    # The label order that the data was generated in, which we want to remap before calculating class accuracies
                    label_order = {0: "Positive", 1: "Neutral", 2: "Negative", 3: "Conflict", 4: "None"}
                else:
                    # Sorted label names returns labels as ["Conflict", "Negative", "Neutral", "None", "Positive"]
                    label_order = {0: "Conflict", 1: "Negative", 2: "Neutral", 3: "None", 4: "Positive"}
                # Remap them so it is easier to calculate class accuracies in SemEval_acc function by stripping away the final
                # {1,2,3} labels for {4,3,2} class accuracies
                label_remap = {"Negative": 0, "Positive": 1, "Neutral": 2, "Conflict": 3, "None": -1}
                label_map = {key: label_remap[value] for key, value in label_order.items()}
                y_pred = np.vectorize(label_map.get)(y_pred)
                y_true = np.vectorize(label_map.get)(y_true)
                scores[:, list(label_map.values())] = scores[:, list(label_map.keys())]
                acc_4_class = SemEval_acc(y_true, scores, n_classes=4)
                acc_3_class = SemEval_acc(y_true, scores, n_classes=3)
                acc_2_class = SemEval_acc(y_true, scores, n_classes=2)
                eval_accuracy = {'2_class': acc_2_class, '3_class': acc_3_class, '4_class': acc_4_class,
                                 'Global': global_acc, 'Aspect': aspect_acc}
            else:
                if task_name[-1] == 'B':
                    # The label order that the data was generated in, which we want to remap before calculating class accuracies
                    label_order = {0: "None", 1: "Positive", 2: "Negative"}
                else:
                    # Sorted label names
                    label_order = {0: "Negative", 1: "None", 2: "Positive"}
                label_remap = {"Negative": 0, "Positive": 1, "None": -1}
                label_map = {key: label_remap[value] for key, value in label_order.items()}
                y_pred = np.vectorize(label_map.get)(y_pred)
                y_true = np.vectorize(label_map.get)(y_true)
                scores[:, list(label_map.values())] = scores[:, list(label_map.keys())]
                aspect_macro_auc, sentiment_acc, sentiment_macro_auc = Sentihood_AUC_Acc(y_true, scores)
                eval_accuracy = {'Strict_Aspect': Sentihood_strict_acc(y_true, y_pred), 'Global': global_acc,
                                 'Aspect': aspect_acc, 'Sentiment': sentiment_acc}
                AUCs = {'Aspect_AUC': aspect_macro_auc, 'Sentiment_AUC': sentiment_macro_auc}
                LOGGER.info(f"AUCs: {AUCs}")
            # Log evaluation accuracy to tensorboard (+1 since not reporting accuracy per batch)
            self.writer.add_scalars('acc', {'_'.join([task_name, acc_name, '_eval_acc']): eval_acc
                                            for acc_name, eval_acc in eval_accuracy.items()},
                                    eval_global_step + 1)

            precision, recall, f1_score = Macro_PRF(y_true, y_pred, task_name)
            LOGGER.info(f"Precision - {precision} | Recall - {recall} | "
                        f"F1 Score - {f1_score} | Accuracies - {eval_accuracy}")

            # Normalise the values we have incremented via batching
            eval_loss = eval_loss / n_eval_steps
            precision_recall_f1_dict = {'Precision': precision, 'Recall': recall, 'F1_Score': f1_score}
            if split_task_name == 'Sentihood':
                metrics_report = {**precision_recall_f1_dict, **AUCs, **eval_accuracy}
            else:
                metrics_report = precision_recall_f1_dict
        else:
            eval_loss, eval_total_correct = 0, 0
            n_eval_steps, n_eval_examples = 0, 0
            y_true = []
            y_pred = []
            if task_name == 'NER':
                mlb = MultiLabelBinarizer(classes=self.label_list['NER'])
            # Evaluate the model with batchwise accuracy like in training
            for batch in tqdm(data_loaders[task_id], "Evaluating"):
                input_ids, segment_ids, input_masks, label_ids = tuple(model_input.to(self.device)
                                                                        for model_input in batch)
                with torch.no_grad():
                    loss, logits = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)

                logits = logits.detach().cpu().numpy()
                if task_name == 'NER':
                    y_true_batch, y_pred_batch = NER_eval(logits, label_ids.to('cpu').numpy())
                else:
                    y_true_batch = label_ids.to('cpu').numpy()
                    y_pred_batch = np.argmax(logits, axis=1)
                y_pred = np.concatenate([y_pred, y_pred_batch])
                y_true = np.concatenate([y_true, y_true_batch])

                # The mean here will calculate the loss averaged over the GPUs.
                eval_loss += loss.mean().item()
                n_eval_steps += 1
                if task_name == 'NER':
                    eval_acc = accuracy_score(mlb.fit_transform(y_true), mlb.fit_transform(y_pred))
                else:
                    eval_acc = accuracy_score(y_true, y_pred)

                # Log the (normalised) batch statistics to tensorboard
                self.writer.add_scalars('loss', {task_name + '_eval_loss': eval_loss / n_eval_steps},
                                        eval_global_step + n_eval_steps)
                self.writer.add_scalars('acc', {task_name + '_eval_acc': eval_acc},
                                        eval_global_step + n_eval_steps)

            # Normalise the values we have incremented via batching
            eval_loss = eval_loss / n_eval_steps
            if task_name == 'NER':
                y_true = mlb.fit_transform(y_true)
                y_pred = mlb.fit_transform(y_pred)
            eval_accuracy = accuracy_score(y_true, y_pred)

            metrics_report = classification_report(y_true, y_pred, target_names=self.label_list[task_name], output_dict=True)
            LOGGER.info('Metrics Report:\n' + classification_report(y_true, y_pred, target_names=self.label_list[task_name]))
            if task_name == 'Streetbees_Mood':
                precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred)
                precision_recall_f1_dict = {'Precision': precision[1], 'Recall': recall[1], 'F1_Score': f1_score[1]}
                metrics_report = {**metrics_report, **precision_recall_f1_dict}

        if return_preds:
            return y_true, y_pred, scores, eval_accuracy, eval_loss, n_eval_steps, metrics_report
        else:
            return eval_accuracy, eval_loss, n_eval_steps, metrics_report

    def test_model(self, return_preds=False):
        # Load the testing data for each task as defined in the config
        # NOTE: label_list only returns the labels in the training and validation set (or predefined labels) as none in test set!
        label_list = {processor.data_name: processor.get_labels() for processor in self.processor_list}
        test_examples = {processor.data_name: processor.get_examples(set_type="test") for processor in self.processor_list}
        max_seq_length = self.config["max_seq_length"]

        test_datasets = []
        # Load in the testing data
        for task_name in self.task_names:
            label_dtype = torch.long if self.task_configs[task_name]['output_type'] == 'CLS' else torch.float
            test_features = convert_examples_to_features(test_examples[task_name], label_list[task_name],
                                                         max_seq_length, self.tokenizer, task_name, self.baseLM_name)
            all_input_ids = torch.tensor([feature.input_ids for feature in test_features], dtype=torch.long)
            all_segment_ids = torch.tensor([feature.segment_ids for feature in test_features], dtype=torch.long)
            all_input_masks = torch.tensor([feature.input_mask for feature in test_features], dtype=torch.long)
            all_label_ids = torch.tensor([feature.label_id for feature in test_features], dtype=label_dtype)
            task_test_data = TensorDataset(all_input_ids, all_segment_ids, all_input_masks, all_label_ids)
            # Load up the test datasets with a relatively large batch size
            test_datasets.append(DataLoader(task_test_data, batch_size=128))

        # Calculate the accuracies, losses and return the predictions if necessary
        test_accs = {task_name: 0 for task_name in self.task_names}
        test_losses = {task_name: 0 for task_name in self.task_names}
        if return_preds:
            test_preds = {task_name: None for task_name in self.task_names}
        for task_id, task_name in enumerate(self.task_names):
            # Pass in eval_global_step = 0, not used in testing as not repeating over many epochs
            if return_preds:
                y_true, test_preds[task_name], scores, test_accs[task_name], test_losses[task_name], _, _ = self.evaluate_model(task_id,
                                                                                                                eval_global_step=0,
                                                                                                                data_loaders=test_datasets,
                                                                                                                return_preds=True)
            else:
                test_accs[task_name], test_losses[task_name], _, _ = self.evaluate_model(task_id, eval_global_step=0,
                                                                                         data_loaders=test_datasets)
        if return_preds:
            return y_true, test_preds, scores, test_accs, test_losses
        else:
            return test_accs, test_losses

if __name__ == '__main__':
    CONFIGS_FOLDER = Path.cwd() / 'configs'
    CONFIG_DIR = CONFIGS_FOLDER / 'run_config.json'

    with open(CONFIG_DIR, 'r') as file_dir:
        CONFIG = json.load(file_dir)

    LOGGER.info(f"You entered config: {CONFIG}")
    MTLModel = MultiTaskLearner(config=CONFIG)
    MTLModel.train()
    MTLModel.test_model()
#    MTLModel.load_data()
 #   MTLModel.evaluate_model(0,0)
