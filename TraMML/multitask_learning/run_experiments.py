# Simple script to automatically run experiments given hyperparameters we want to test
import itertools
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from learners import MultiTaskLearner
from data_processing import TASK_PRIORITIES

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO)
LOGGER = logging.getLogger(__name__)

RUNS_FOLDER = Path.cwd() / 'runs'
EXPERIMENT_LOG = RUNS_FOLDER / 'experiment_log.csv'

# TODO: Have a think about which hyperparameters to test (cf. MultiTaskLearning)
hparams_to_test = {'tasks': ['SST-2', 'SemEval_QA_M', 'SST-2, SemEval_QA_M', 'IMDB', 'SST-2, IMDB'],
                   'sampling_mode': ['sequential', 'random', 'prop', 'sqrt', 'square', 'anneal'],
                   'model_name': ['bert-base-cased', 'xlnet-base-cased']
                    }
metrics_to_report = ['final_loss_train', 'final_loss_dev', 'final_loss_test', 'final_acc_dev', 'final_acc_test'] + \
                    ["_".join(['SemEval_final_acc_dev', str(n), "class"]) for n in [2, 3, 4]] + \
                    ["_".join(['SemEval_final_acc_test', str(n), "class"]) for n in [2, 3, 4]] + \
                    ['Sentihood_strict__aspect_acc_dev', 'Sentihood_strict_aspect_acc_test', 'Sentihood_Sentiment_AUC', 'Sentihood_Aspect_AUC'] + \
                    ['Precision', 'Recall', 'F1_Score']


class ExperimentRunner:
    def __init__(self, save_dir, hparams_to_test=hparams_to_test):
        # Create a cleaned hyperparameter list
        self.hp_list = ExperimentRunner.hp_list_cleaner(itertools.product(*hparams_to_test.values()))

        # Create an experiment log and folders if not already present
        self.root_dir = Path(save_dir)
        self.saved_models_folder = self.root_dir / 'saved_models'
        self.runs_folder = self.root_dir / 'runs'
        self.reports_folder = self.root_dir / 'reports'
        self.experiment_log_path = self.root_dir / 'experiment_log.csv'
        if not self.root_dir.is_dir():
            self.root_dir.mkdir()
        if not self.runs_folder.is_dir():
            self.runs_folder.mkdir()
        if not self.reports_folder.is_dir():
            self.reports_folder.mkdir()
        if not self.saved_models_folder.is_dir():
            self.saved_models_folder.mkdir()
        if not self.experiment_log_path.is_file():
            log_cols = metrics_to_report + ['Done?', 'UniqueID']
            experiment_log_df = ExperimentRunner.create_experiment_log(self.hp_list, experiment_log_columns=log_cols)
            experiment_log_df.to_csv(self.experiment_log_path)
        self.experiment_log_df = pd.read_csv(self.experiment_log_path, index_col=[0,1,2,3])

        # Check which experiments have already been run
        self.experiments_already_run = self.experiment_log_df['UniqueID'][self.experiment_log_df['Done?'] == 'Yes'].values
        LOGGER.info(f'Already run experiments: {self.experiments_already_run}')

    def run(self, save_best_f1_model_name=None, save_all_models=False):
        """
        Runs the Experiments defined by the experiment config

        Parameters
        ----------
        save_best_acc_model_name : str or None, optional
            String of model acc to store the best checkpoint of or None, by default 'SemEval_QA_M'
        save_all_models : bool, optional
            Whether or not to save each model in the experiment, by default False
        """
        num_experiments_so_far = sum(self.experiment_log_df['Done?'].notna())
        total_experiments_to_run = len(self.experiment_log_df)
        if save_best_f1_model_name:
            if num_experiments_so_far == 0:
                best_f1 = 0
            else:
                acc_col = 'final_acc_test' if save_best_acc_model_name.split('_')[0] != 'SemEval' else 'SemEval_final_acc_test'
                f1_col = 'F1_Score'
                best_acc = self.experiment_log_df[acc_col].max()
                best_f1 = self.experiment_log_df[f1_col].max()

        for tasks, sampling_mode, model_name in self.hp_list:
            unique_experiment_id = "|".join([model_name, tasks, sampling_mode])
            # Allow us to pick up where we left off by skipping over experiments we have already run
            if unique_experiment_id in self.experiments_already_run:
                continue

            # Perform Experiments for the given setting of hyperparameters
            # If hyperparameter not specified, the default will be used (see MultiTaskLearner class)
            run_config = {'data_dir': '../../../datascience-projects/internal/multitask_learning/processed_data',
                          'log_dir': str(self.runs_folder / unique_experiment_id),
                          'model_name': model_name,
                          'sampling_mode': sampling_mode,
                          'tasks': tasks}
            LOGGER.info(f"Running experiment with run config: {run_config}")
            MTLearner = MultiTaskLearner(config=run_config)
            final_loss_train, final_loss_dev, final_acc_dev, final_metrics_report = MTLearner.train()
            final_acc_test, final_loss_test = MTLearner.test_model()
            LOGGER.info(f'At test time, the results are: Accuracy - {final_acc_test}')

            # Log the experiment to csv and save the final metrics report
            metrics_to_log_tuple = final_loss_train, final_loss_dev, final_acc_dev, final_acc_test, final_loss_test, final_metrics_report
            self.log_experiment(unique_experiment_id, metrics_to_log_tuple)

            save_name = unique_experiment_id.replace(", ", "|")
            pd.DataFrame(final_metrics_report).T.to_csv(str(self.reports_folder) +'/' + save_name + ".csv")
            with open(str(self.reports_folder) + '/' + save_name + '.json', 'w') as save_path:
                json.dump(final_metrics_report, save_path)

            # Save models
            if save_all_models:
                save_name += '.pt'
                torch.save(MTLearner.model.state_dict(), self.saved_models_folder / save_name)

            if save_best_f1_model_name and final_metrics_report[save_best_f1_model_name]['F1_Score'] > best_f1:
                best_f1 = final_metrics_report[save_best_f1_model_name]['F1_Score']
                save_name = 'best_model_' + save_best_f1_model_name + '.pt'
                torch.save(MTLearner.model.state_dict(), self.saved_models_folder / save_name)

            num_experiments_so_far += 1
            LOGGER.info(f'Completed {num_experiments_so_far}/{total_experiments_to_run} experiments!')

    @staticmethod
    def hp_list_cleaner(hp_list):
        """
        A helper function for cleaning the hyperparameter list of tuples

        Parameters
        ----------
        hp_list : list of tuples
            List of hyperparameter tuples

        Returns
        -------
        list of tuples
            The cleaned list of hyperparameter tuples
        """
        hp_list_cleaned = []
        for hp_tuple in hp_list:
            tasks, sampling_mode, model_name = hp_tuple

            single_task_list = tasks.split(", ")
            num_tasks = len(single_task_list)

            # If we have 1 task, then all sampling modes == sequential
            if num_tasks == 1 and sampling_mode != 'sequential':
                continue
            # If we have 2 tasks of the same priority, then only 2 unique sampling modes: sequential & random
            if (num_tasks == 2 and
                    (TASK_PRIORITIES[single_task_list[0]] == TASK_PRIORITIES[single_task_list[1]]) and
                    sampling_mode not in ['sequential', 'random']):
                continue
            else:
                hp_list_cleaned.append(hp_tuple)
        return hp_list_cleaned

    @staticmethod
    def create_experiment_log(hp_list_cleaned, experiment_log_columns):
        """
        Create the experiment log using the cleaned hyperparameter list and any additional
        columns we want to add

        Parameters
        ----------
        hp_list_cleaned : list of tuples
            List of hyperparameter tuples (cleaned) that we want to test
        experiment_log_columns : list of strings
            List of additional columns we want to have, typically the metrics_to_report plus
            a 'Done?' column

        Returns
        -------
        pandas dataframe
            Experiment log
        """
        task_tuples = []
        for multi_task, sampling_mode, model_name in hp_list_cleaned:
            for single_task in multi_task.split(", "):
                task_tuples += [(model_name, single_task, multi_task, sampling_mode)]
        multi_index = pd.MultiIndex.from_tuples(task_tuples, names = ['Model Name', 'Single Task', 'Tasks', 'Sampling Mode'])
        experiment_log_df = pd.DataFrame(columns=experiment_log_columns, index=multi_index)
        # Create a unique index based on the multitasks and any hyperparameters fed in (to the multiindex) except 'single task' which is idx[1]
        experiment_log_df['UniqueID'] = experiment_log_df.index.to_series().apply(lambda idx: "|".join(setting for setting in [idx[0]] + list(idx[2:])))
        return experiment_log_df

    def log_experiment(self, unique_experiment_id, metrics_to_log_tuple):
        """
        Logs a specific experiment by saving the metrics to a csv

        Parameters
        ----------
        unique_experiment_id : str
            A string corresponding to a specific experiment to be run
        metrics_to_log_tuple : tuple
            A tuple of metrics_to_log
        """
        final_loss_train, final_loss_dev, final_acc_dev, final_acc_test, final_loss_test, final_metrics_report = metrics_to_log_tuple
        model_name, tasks, sampling_mode = unique_experiment_id.split('|')
        # Mark the experiment as done in the log (All metrics etc stored in the tensorboard log not in csv)
        for single_task in tasks.split(", "):
            task_df = self.experiment_log_df.loc[model_name, single_task, tasks, sampling_mode]
            if single_task.split("_")[0] == 'SemEval':
                single_task_results = pd.Series({'final_loss_train': final_loss_train[single_task],
                                                 'final_loss_dev': final_loss_dev[single_task],
                                                 'final_loss_test': final_loss_dev[single_task],
                                                 'final_acc_dev': final_acc_dev[single_task]['Aspect'],
                                                 'final_acc_test': final_acc_test[single_task]['Aspect'],
                                                 'SemEval_final_acc_dev_2_class': final_acc_dev[single_task]['2_class'],
                                                 'SemEval_final_acc_dev_3_class': final_acc_dev[single_task]['3_class'],
                                                 'SemEval_final_acc_dev_4_class': final_acc_dev[single_task]['4_class'],
                                                 'SemEval_final_acc_test_2_class': final_acc_test[single_task]['2_class'],
                                                 'SemEval_final_acc_test_3_class': final_acc_test[single_task]['3_class'],
                                                 'SemEval_final_acc_test_4_class': final_acc_test[single_task]['4_class'],
                                                 'Precision': final_metrics_report[single_task]['Precision'],
                                                 'Recall': final_metrics_report[single_task]['Recall'],
                                                 'F1_Score': final_metrics_report[single_task]['F1_Score'],
                                                 'Done?': 'Yes'})
            elif single_task.split("_")[0] == 'Sentihood':
                single_task_results = pd.Series({'final_loss_train': final_loss_train[single_task],
                                                 'final_loss_dev': final_loss_dev[single_task],
                                                 'final_loss_test': final_loss_dev[single_task],
                                                 'final_acc_dev': final_acc_dev[single_task]['Sentiment'],
                                                 'final_acc_test': final_acc_test[single_task]['Sentiment'],
                                                 'Sentihood_strict_aspect_acc_dev': final_acc_dev[single_task]['Strict_Aspect'],
                                                 'Sentihood_strict_aspect_acc_test': final_acc_dev[single_task]['Strict_Aspect'],
                                                 'Sentihood_Sentiment_AUC': final_metrics_report[single_task]['Sentiment_AUC'],
                                                 'Sentihood_Aspect_AUC': final_metrics_report[single_task]['Aspect_AUC'],
                                                 'Precision': final_metrics_report[single_task]['Precision'],
                                                 'Recall': final_metrics_report[single_task]['Recall'],
                                                 'F1_Score': final_metrics_report[single_task]['F1_Score'],
                                                 'Done?': 'Yes'})
            elif single_task == 'Streetbees_Mood':
                single_task_results = pd.Series({'final_loss_train': final_loss_train[single_task],
                                                 'final_loss_dev': final_loss_dev[single_task],
                                                 'final_loss_test': final_loss_dev[single_task],
                                                 'final_acc_dev': final_acc_dev[single_task],
                                                 'final_acc_test': final_acc_test[single_task],
                                                 'Precision': final_metrics_report[single_task]['Precision'],
                                                 'Recall': final_metrics_report[single_task]['Recall'],
                                                 'F1_Score': final_metrics_report[single_task]['F1_Score'],
                                                 'Done?': 'Yes'})
            else:
                single_task_results = pd.Series({'final_loss_train': final_loss_train[single_task],
                                                 'final_loss_dev': final_loss_dev[single_task],
                                                 'final_loss_test': final_loss_dev[single_task],
                                                 'final_acc_dev': final_acc_dev[single_task],
                                                 'final_acc_test': final_acc_test[single_task],
                                                 'Done?': 'Yes'})
            task_df.update(single_task_results)
            self.experiment_log_df.loc[model_name, single_task, tasks, sampling_mode] = task_df
        self.experiment_log_df.to_csv(self.experiment_log_path)


if __name__ == '__main__':
    runner = ExperimentRunner()
    runner.run()
