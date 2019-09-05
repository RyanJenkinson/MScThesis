import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import random
import json
import os
import argparse
from pathlib import Path
from copy import deepcopy
import pdb
## TENSORBOARD LOGGING ##
from tensorboardX import SummaryWriter

from modelling import MultiTaskModel
from utils_meta import MetaLearningDataset

class MetaLearner(nn.Module):

    def __init__(self, model, args):

        super(MetaLearner, self).__init__()
        # This will be our language model
        self.model = model
        # TODO: Check the first argument i.e number of train steps
        self.model_optim, self.scheduler = self.model.prepare_optimizer_and_scheduler(args.num_updates*args.num_epochs,
                                                                                      learning_rate=args.update_lr)

        self.args = args

        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.num_updates = args.num_updates
        self.test_size = args.K
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.meta_optim = optim.Adam(self.model.parameters(), lr=self.meta_lr)

    def inner_train_step(self, batch, task_name):
        input_ids, segment_ids, input_masks, label_ids = tuple(model_input.to(self.device)
                                                                   for model_input in batch)
        loss, logits = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)
        loss.backward()
        self.scheduler.step()  # Update learning rate scheduler
        self.model_optim.step()  # Update optimizer
        self.model.zero_grad()
        return loss, logits

    def forward(self, task_name, train_batches, test_batches, tbx, num_tensorboard_steps, evaluate):
        # x_train: [num tasks, train size, MAX LENGTH]
        # x_test: [num_tasks, test size, MAX LENGTH]
        # train size = test size = K

        losses = [0 for _ in range(self.num_updates + 1)]
        corrects = [0 for _ in range(self.num_updates + 1)]

        self.model.zero_grad()
        # stored_weights = list(p.data for p in self.model.parameters())
        original_weights = deepcopy(self.model.state_dict())
        new_weights = None
        fast_weights = {name: 0 for name in original_weights}
        for i in range(len(train_batches)):
            train_batch = tuple(inp[i] for inp in train_batches.values())
            # Run model on training data. Each batch is a dictionary so loop over the values
            loss, _ = self.inner_train_step(train_batch, task_name)
            new_weights = deepcopy(self.model.state_dict())
            ## tb records loss ##
            loss_val = loss.mean().item()
            tbx.add_scalar('train/loss', loss_val, num_tensorboard_steps)

            # evaluate on test data before gradient update
            # self.model.load_state_dict({ name: original_weights[name] for name in original_weights })
            self.model.load_state_dict(original_weights)
            with torch.no_grad():
                # set size * 2 (binary)
                batch = tuple(inp[i] for inp in test_batches.values())
                input_ids, segment_ids, input_masks, label_ids = tuple(model_input.to(self.device)
                                                                   for model_input in batch)
                loss, logits = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)
                losses[0] += loss.mean().item()

                pred = F.softmax(logits, dim=1).argmax(dim=1)
                correct = torch.eq(pred, label_ids).sum().item()
                corrects[0] += correct


            # update weights
            self.model.load_state_dict(new_weights)
            # self.model.load_state_dict({name: new_weights[-1][name] for name in new_weights})
            # evaluate on test data after gradient update
            with torch.no_grad():
                loss, logits = self.model(input_ids, segment_ids, input_masks, task_name, label_ids)
                losses[1] += loss.mean().item()

                pred = F.softmax(logits, dim=1).argmax(dim=1)
                correct = torch.eq(pred, label_ids).sum().item()
                corrects[1] += correct

            # restore original model weights
            self.model.load_state_dict(original_weights)
            # self.model.load_state_dict({ name: original_weights[name] for name in original_weights })

            # Update running average of new weights
            fast_weights = {name: (i*fast_weights[name] + new_weights[name]) / (i+1) for name in original_weights}
#        loss = losses[-1] / len(test_batches)
#        loss = Variable(loss, requires_grad=True)

#        self.meta_optim.zero_grad()

        # meta learning step
        if not evaluate:
            # loss.backward()
            # self.meta_optim.step()
#            ws = len(new_weights)
 #           fast_weights = { name : new_weights[0][name]/float(ws) for name in new_weights[0] }
  #          for i in range(1, ws):
   #             for name in new_weights[i]:
    #                fast_weights[name] += new_weights[i][name]/float(ws)

            self.model.load_state_dict({name :
                original_weights[name] + ((fast_weights[name] - original_weights[name]) * self.meta_lr) for name in original_weights})

        losses = np.array(losses) / (self.test_size * self.args.batch_size)
        accs = np.array(corrects) / (self.test_size * self.args.batch_size)

        ## tb records loss ##
        before_loss_val = losses[0]
        tbx.add_scalar('test/before_gradient_update/loss', before_loss_val, num_tensorboard_steps)
        before_acc = accs[0]
        tbx.add_scalar('test/before_gradient_update/acc', before_acc, num_tensorboard_steps)

        after_loss_val = losses[1]
        tbx.add_scalar('test/after_gradient_update/loss', after_loss_val, num_tensorboard_steps)
        after_acc = accs[1]
        tbx.add_scalar('test/after_gradient_update/acc', after_acc, num_tensorboard_steps)
        return losses, accs

def set_seed(seed):
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    ### INIT TB LOGGING ###
    save_dir = Path.cwd() / 'few_shot'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tbx = SummaryWriter(str(save_dir))

    # CONFIGS_FOLDER = Path.cwd() / 'configs'
    # CONFIG_DIR = CONFIGS_FOLDER / 'meta_run_config.json'

    # with open(CONFIG_DIR, 'r') as file_dir:
    #     CONFIG = json.load(file_dir)

    # LOGGER.info(f"You entered config: {CONFIG}")

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')

    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=8) # num tasks per batch
    parser.add_argument('--K', type=int, default=5) # K-shot learning
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--meta_lr', type=float, default=1e-3)
    parser.add_argument('--update_lr', type=float, default=0.1)
    parser.add_argument('--num_updates', type=int, default=1)

    args = parser.parse_args()
    set_seed(args.seed)
    task_name = 'Streetbees_Mood'
    task_configs = {'Streetbees_Mood': {"num_labels": 2,
                                        "output_type": "CLS"}}
    model = MultiTaskModel(task_configs=task_configs, model_name_or_config=args.model_name)
    meta_learner = MetaLearner(model, args)

    train_dataset = MetaLearningDataset(split='train', args=args)
    test_dataset = MetaLearningDataset(split='test', args=args)

    print('There are', train_dataset.num_tasks, 'training tasks:', train_dataset.task_names)
    print('There are', test_dataset.num_tasks, 'test tasks:', test_dataset.task_names)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.num_tasks)

    for epoch in range(1, args.num_epochs+1):
        num_tensorboard_steps = 0
        print('EPOCH %d' % epoch)
        print('TRAIN (batch size = %d)' % args.batch_size)
        for batch_idx, batch in enumerate(train_loader):
            num_tensorboard_steps += 1
            # sample without replacement from same task for train and test (K of each)
            train_data, test_data = batch

            # train the metalearner
            losses, accs = meta_learner.forward(task_name, train_data, test_data, tbx, num_tensorboard_steps, evaluate=False)
            print(losses, accs)

        print('TEST (batch size = %d)' % test_dataset.num_tasks)
        for batch_idx, batch in enumerate(test_loader):
            train_data, test_data = batch

            # test the metalearner
            losses, accs = meta_learner.forward(task_name, train_data, test_data, tbx, num_tensorboard_steps, evaluate=True)
            print(losses, accs)
