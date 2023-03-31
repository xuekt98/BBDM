import argparse
import datetime
import pdb
import random
import time

import yaml
import os
import traceback

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm

from runners.base.EMA import EMA
from runners.utils import make_save_dirs, make_dir, get_dataset, remove_file, get_dataset_list


class BaseRunner(ABC):
    def __init__(self, config):
        self.net = None  # Neural Network
        self.optimizer = None  # optimizer
        self.scheduler = None  # scheduler
        self.config = config  # config from configuration file

        # set training params
        self.global_epoch = 0  # global epoch
        if config.args.sample_at_start:
            self.global_step = -1  # global step
        else:
            self.global_step = 0

        self.GAN_buffer = {}  # GAN buffer for Generative Adversarial Network
        self.topk_checkpoints = {}  # Top K checkpoints

        # set log and save destination
        self.config.result = argparse.Namespace()
        self.config.result.image_path, \
            self.config.result.ckpt_path, \
            self.config.result.log_path, \
            self.config.result.sample_path, \
            self.config.result.sample_to_eval_path = make_save_dirs(self.config.args,
                                                                    prefix=self.config.task_name,
                                                                    suffix=self.config.model.model_name)

        self.save_config()  # save configuration file
        self.writer = SummaryWriter(self.config.result.log_path)  # initialize SummaryWriter

        # initialize model
        self.net, self.optimizer, self.scheduler = self.initialize_model_optimizer_scheduler(self.config)

        self.print_model_summary(self.net)

        # initialize EMA
        self.use_ema = False if not self.config.model.__contains__('EMA') else self.config.model.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(self.config.model.EMA.ema_decay)
            self.update_ema_interval = self.config.model.EMA.update_ema_interval
            self.start_ema_step = self.config.model.EMA.start_ema_step
            self.ema.register(self.net)

        # load model from checkpoint
        self.load_model_from_checkpoint()

        # initialize DDP
        if self.config.training.use_DDP:
            self.net = DDP(self.net, device_ids=[self.config.training.local_rank],
                           output_device=self.config.training.local_rank)
        else:
            self.net = self.net.to(self.config.training.device[0])
        # self.ema.reset_device(self.net)

    # save configuration file
    def save_config(self):
        save_path = os.path.join(self.config.result.ckpt_path, 'config.yaml')
        save_config = self.config
        with open(save_path, 'w') as f:
            yaml.dump(save_config, f)

    def initialize_model_optimizer_scheduler(self, config, is_test=False):
        """
        get model, optimizer, scheduler
        :param args: args
        :param config: config
        :param is_test: is_test
        :return: net: Neural Network, nn.Module;
                 optimizer: a list of optimizers;
                 scheduler: a list of schedulers or None;
        """
        net = self.initialize_model(config)
        optimizer, scheduler = None, None
        if not is_test:
            optimizer, scheduler = self.initialize_optimizer_scheduler(net, config)
        return net, optimizer, scheduler

    # load model, EMA, optimizer, scheduler from checkpoint
    def load_model_from_checkpoint(self):
        model_states = None
        if self.config.model.__contains__('model_load_path') and self.config.model.model_load_path is not None:
            print(f"load model {self.config.model.model_name} from {self.config.model.model_load_path}")
            model_states = torch.load(self.config.model.model_load_path, map_location='cpu')

            self.global_epoch = model_states['epoch']
            self.global_step = model_states['step']

            # load model
            self.net.load_state_dict(model_states['model'])

            # load ema
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.net)

            # load optimizer and scheduler
            if self.config.args.train:
                if self.config.model.__contains__(
                        'optim_sche_load_path') and self.config.model.optim_sche_load_path is not None:
                    optimizer_scheduler_states = torch.load(self.config.model.optim_sche_load_path, map_location='cpu')
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(optimizer_scheduler_states['optimizer'][i])

                    if self.scheduler is not None:
                        for i in range(len(self.optimizer)):
                            self.scheduler[i].load_state_dict(optimizer_scheduler_states['scheduler'][i])
        return model_states

    def get_checkpoint_states(self, stage='epoch_end'):
        optimizer_state = []
        for i in range(len(self.optimizer)):
            optimizer_state.append(self.optimizer[i].state_dict())

        scheduler_state = []
        for i in range(len(self.scheduler)):
            scheduler_state.append(self.scheduler[i].state_dict())

        optimizer_scheduler_states = {
            'optimizer': optimizer_state,
            'scheduler': scheduler_state
        }

        model_states = {
            'step': self.global_step,
        }

        if self.config.training.use_DDP:
            model_states['model'] = self.net.module.state_dict()
        else:
            model_states['model'] = self.net.state_dict()

        if stage == 'exception':
            model_states['epoch'] = self.global_epoch
        else:
            model_states['epoch'] = self.global_epoch + 1

        if self.use_ema:
            model_states['ema'] = self.ema.shadow
        return model_states, optimizer_scheduler_states

    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        if self.config.training.use_DDP:
            self.ema.update(self.net.module, with_decay=with_decay)
        else:
            self.ema.update(self.net, with_decay=with_decay)

    def apply_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.apply_shadow(self.net.module)
            else:
                self.ema.apply_shadow(self.net)

    def restore_ema(self):
        if self.use_ema:
            if self.config.training.use_DDP:
                self.ema.restore(self.net.module)
            else:
                self.ema.restore(self.net)

    # Evaluation and sample part
    @torch.no_grad()
    def validation_step(self, val_batch, epoch, step):
        self.apply_ema()
        self.net.eval()
        loss = self.loss_fn(net=self.net,
                            batch=val_batch,
                            epoch=epoch,
                            step=step,
                            opt_idx=0,
                            stage='val_step')
        if len(self.optimizer) > 1:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=1,
                                stage='val_step')
        self.restore_ema()

    @torch.no_grad()
    def validation_epoch(self, val_loader, epoch, task_index=None):
        self.apply_ema()
        self.net.eval()

        task_index = "" if task_index is None else task_index
        pbar = tqdm(val_loader, total=len(val_loader), smoothing=0.01)
        step = 0
        loss_sum = 0.
        dloss_sum = 0.
        for val_batch in pbar:
            loss = self.loss_fn(net=self.net,
                                batch=val_batch,
                                epoch=epoch,
                                step=step,
                                opt_idx=0,
                                stage='val',
                                write=False)
            loss_sum += loss
            if len(self.optimizer) > 1:
                loss = self.loss_fn(net=self.net,
                                    batch=val_batch,
                                    epoch=epoch,
                                    step=step,
                                    opt_idx=1,
                                    stage='val',
                                    write=False)
                dloss_sum += loss
            step += 1
        average_loss = loss_sum / step
        self.writer.add_scalar(f'val_epoch/loss{task_index}', average_loss, epoch)
        if len(self.optimizer) > 1:
            average_dloss = dloss_sum / step
            self.writer.add_scalar(f'val_dloss_epoch/loss{task_index}', average_dloss, epoch)
        self.restore_ema()
        return average_loss

    @torch.no_grad()
    def sample_step(self, data_batch, stage='train', task_index=None):
        self.apply_ema()
        self.net.eval()
        sample_path = make_dir(os.path.join(self.config.result.image_path, str(self.global_step)))
        if self.config.training.use_DDP:
            self.sample(self.net.module, data_batch, sample_path, stage=stage, task_index=task_index)
        else:
            self.sample(self.net, data_batch, sample_path, stage=stage, task_index=task_index)
        self.restore_ema()

    # abstract methods
    @abstractmethod
    def print_model_summary(self, net):
        pass

    @abstractmethod
    def initialize_model(self, config):
        """
        initialize model
        :param config: config
        :return: nn.Module
        """
        pass

    @abstractmethod
    def initialize_optimizer_scheduler(self, net, config):
        """
        initialize optimizer and scheduler
        :param net: nn.Module
        :param config: config
        :return: a list of optimizers; a list of schedulers
        """
        pass

    @abstractmethod
    def loss_fn(self, net, batch, epoch, step, opt_idx=0, stage='train', write=True):
        """
        loss function
        :param net: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass

    @abstractmethod
    def sample(self, net, batch, sample_path, stage='train', task_index=None):
        """
        sample a single batch
        :param task_index: multi task index
        :param net: nn.Module
        :param batch: batch
        :param sample_path: path to save samples
        :param stage: train, val, test
        :return:
        """
        pass

    @abstractmethod
    def sample_to_eval(self, net, test_loader, sample_path):
        """
        sample among the test dataset to calculate evaluation metrics
        :param net: nn.Module
        :param test_loader: test dataloader
        :param sample_path: path to save samples
        :return:
        """
        pass

    def on_save_checkpoint(self, net, train_loader, val_loader, epoch, step):
        """
        additional operations whilst saving checkpoint
        :param net: nn.Module
        :param train_loader: train data loader
        :param val_loader: val data loader
        :param epoch: epoch
        :param step: step
        :return:
        """
        pass

    def reload_data(self):
        train_dataset_list, val_dataset_list, test_dataset_list = get_dataset_list(self.config.data)
        train_loader_list, val_loader_list, test_loader_list = [], [], []
        for i in range(len(train_dataset_list)):
            if self.config.training.use_DDP:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_list[i])
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_list[i])
                test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset_list[i])
                train_loader_list.append(DataLoader(train_dataset_list[i],
                                                    batch_size=self.config.data.train.batch_size,
                                                    num_workers=8,
                                                    drop_last=True,
                                                    sampler=train_sampler))
                val_loader_list.append(DataLoader(val_dataset_list[i],
                                                  batch_size=self.config.data.val.batch_size,
                                                  num_workers=8,
                                                  drop_last=True,
                                                  sampler=val_sampler))
                test_loader_list.append(DataLoader(test_dataset_list[i],
                                                   batch_size=self.config.data.test.batch_size,
                                                   shuffle=False,
                                                   num_workers=1,
                                                   drop_last=True,
                                                   sampler=test_sampler))
            else:
                train_loader_list.append(DataLoader(train_dataset_list[i],
                                                    batch_size=self.config.data.train.batch_size,
                                                    shuffle=self.config.data.train.shuffle,
                                                    num_workers=8,
                                                    drop_last=True))
                val_loader_list.append(DataLoader(val_dataset_list[i],
                                                  batch_size=self.config.data.val.batch_size,
                                                  shuffle=self.config.data.val.shuffle,
                                                  num_workers=8,
                                                  drop_last=True))
                test_loader_list.append(DataLoader(test_dataset_list[i],
                                                   batch_size=self.config.data.test.batch_size,
                                                   shuffle=False,
                                                   num_workers=1,
                                                   drop_last=True))
        return train_loader_list, val_loader_list, test_loader_list

    def train(self):
        print(self.__class__.__name__)
        print(f"start training {self.config.model.model_name} "
              f"on {self.config.data.dataset_name_list}, {self.config.training.n_steps} steps")

        try:
            # accumulate_grad_batch_count = 0
            # load data
            train_loader_list, val_loader_list, _ = self.reload_data()
            train_iter_list, val_iter_list, test_iter_list = [], [], []
            for i in range(len(train_loader_list)):
                train_iter_list.append(iter(train_loader_list[i]))
                val_iter_list.append(iter(val_loader_list[i]))

            start_step = self.global_step
            n_tasks = len(train_loader_list)

            start_time = time.time()
            pbar = tqdm(range(start_step, self.config.training.n_steps))
            for step in pbar:
                # reload data
                if step % self.config.training.reload_data_interval == 0:
                    self.global_epoch += 1
                    train_loader_list, val_loader_list, _ = self.reload_data()
                    for i in range(n_tasks):
                        train_iter_list[i] = iter(train_loader_list[i])
                        val_iter_list[i] = iter(val_loader_list[i])

                    end_time = time.time()
                    elapsed_rounded = int(round((end_time - start_time)))
                    print(f"training {self.global_step - start_step} steps time: "
                          + str(datetime.timedelta(seconds=elapsed_rounded)))

                # get batch
                try:
                    train_batch = next(train_iter_list[step % n_tasks])
                except StopIteration:
                    train_iter_list[step % n_tasks] = iter(train_loader_list[step % n_tasks])
                    train_batch = next(train_iter_list[step % n_tasks])

                self.global_step = step
                self.net.train()

                losses = []
                for i in range(len(self.optimizer)):
                    loss = self.loss_fn(net=self.net,
                                        batch=train_batch,
                                        epoch=self.global_epoch,
                                        step=self.global_step,
                                        opt_idx=i,
                                        stage='train')
                    loss.backward()
                    if self.global_step % self.config.training.accumulate_grad_batches == 0:
                        self.optimizer[i].step()
                        self.optimizer[i].zero_grad()

                    losses.append(loss.detach().mean())
                    if self.scheduler is not None:
                        self.scheduler[i].step(loss)

                if self.use_ema and self.global_step % \
                        (self.update_ema_interval * self.config.training.accumulate_grad_batches) == 0:
                    self.step_ema()

                if len(self.optimizer) > 1:
                    pbar.set_description(
                        f'iter: {self.global_step} loss-1: {losses[0]:.4f} loss-2: {losses[1]:.4f}'
                    )
                else:
                    pbar.set_description(
                        f'iter: {self.global_step} loss: {losses[0]:.4f}'
                    )

                with torch.no_grad():
                    if self.global_step % 50 == 0:
                        val_batch = next(iter(val_loader_list[random.randint(0, n_tasks - 1)]))
                        self.validation_step(val_batch=val_batch, epoch=self.global_epoch, step=self.global_step)

                    if self.global_step % int(self.config.training.sample_interval) == 0:
                        if not self.config.training.use_DDP or \
                                (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                            for i in range(n_tasks):
                                val_batch = next(iter(val_loader_list[i]))
                                self.sample_step(data_batch=val_batch, stage='val', task_index=i)
                            self.sample_step(data_batch=train_batch, stage='train')
                            torch.cuda.empty_cache()

                # validation
                if self.global_step % self.config.training.validation_interval == 0 or \
                        self.global_step == self.config.training.n_steps - 1:
                    if not self.config.training.use_DDP or \
                            (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                        with torch.no_grad():
                            print("validating epoch...")
                            total_loss = 0
                            for i in range(n_tasks):
                                total_loss += self.validation_epoch(val_loader_list[i], self.global_epoch)
                            average_loss = total_loss / n_tasks
                            torch.cuda.empty_cache()
                            print("validating epoch success")

                # save checkpoint
                if self.global_step % self.config.training.save_interval == 0 or \
                        self.global_step == self.config.training.n_steps - 1:
                    if not self.config.training.use_DDP or \
                            (self.config.training.use_DDP and self.config.training.local_rank) == 0:
                        with torch.no_grad():
                            print("saving latest checkpoint...")
                            for i in range(n_tasks):
                                self.on_save_checkpoint(self.net, train_loader_list[i], val_loader_list[i],
                                                        self.global_epoch, self.global_step)
                            model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='epoch_end')

                            # remove old checkpoints
                            temp = 0
                            while temp < self.global_step:
                                remove_file(os.path.join(self.config.result.ckpt_path, f'latest_model_{temp}.pth'))
                                remove_file(
                                    os.path.join(self.config.result.ckpt_path, f'latest_optim_sche_{temp}.pth'))
                                temp += 1

                            # save latest checkpoint
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_model_{self.global_step}.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'latest_optim_sche_{self.global_step}.pth'))
                            torch.save(model_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_model.pth'))
                            torch.save(optimizer_scheduler_states,
                                       os.path.join(self.config.result.ckpt_path,
                                                    f'last_optim_sche.pth'))

                            if self.config.args.save_top:
                                # save top_k checkpoints
                                model_ckpt_name = os.path.join(self.config.result.ckpt_path,
                                                               f'model_checkpoint_{average_loss:.2f}_'
                                                               f'step={self.global_step}.pth')
                                optim_sche_ckpt_name = os.path.join(self.config.result.ckpt_path,
                                                                    f'optim_sche_checkpoint_{average_loss:.2f}_'
                                                                    f'step={self.global_step}.pth')

                                top_key = 'top'
                                if top_key not in self.topk_checkpoints:
                                    self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                      'model_ckpt_name': model_ckpt_name,
                                                                      'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                    print(f"saving top checkpoint: average_loss={average_loss} step={self.global_step}")
                                    torch.save(model_states,
                                               model_ckpt_name,
                                               _use_new_zipfile_serialization=False)
                                    torch.save(optimizer_scheduler_states,
                                               optim_sche_ckpt_name,
                                               _use_new_zipfile_serialization=False)
                                else:
                                    if average_loss < self.topk_checkpoints[top_key]["loss"]:
                                        print("remove " + self.topk_checkpoints[top_key]["model_ckpt_name"])
                                        remove_file(self.topk_checkpoints[top_key]['model_ckpt_name'])
                                        remove_file(self.topk_checkpoints[top_key]['optim_sche_ckpt_name'])

                                        print(
                                            f"saving top checkpoint: average_loss={average_loss} "
                                            f"step={self.global_step}")

                                        self.topk_checkpoints[top_key] = {"loss": average_loss,
                                                                          'model_ckpt_name': model_ckpt_name,
                                                                          'optim_sche_ckpt_name': optim_sche_ckpt_name}

                                        torch.save(model_states,
                                                   model_ckpt_name,
                                                   _use_new_zipfile_serialization=False)
                                        torch.save(optimizer_scheduler_states,
                                                   optim_sche_ckpt_name,
                                                   _use_new_zipfile_serialization=False)

        except BaseException as e:
            if not self.config.training.use_DDP or (
                    self.config.training.use_DDP and self.config.training.local_rank) == 0:
                print("exception save model start....")
                print(self.__class__.__name__)
                model_states, optimizer_scheduler_states = self.get_checkpoint_states(stage='exception')
                torch.save(model_states,
                           os.path.join(self.config.result.ckpt_path, f'last_model.pth'),
                           _use_new_zipfile_serialization=False)
                torch.save(optimizer_scheduler_states,
                           os.path.join(self.config.result.ckpt_path, f'last_optim_sche.pth'),
                           _use_new_zipfile_serialization=False)

                print("exception save model success!")

            print('str(Exception):\t', str(Exception))
            print('str(e):\t\t', str(e))
            print('repr(e):\t', repr(e))
            print('traceback.print_exc():')
            traceback.print_exc()
            print('traceback.format_exc():\n%s' % traceback.format_exc())

    @torch.no_grad()
    def test(self):
        train_loader_list, val_loader_list, test_loader_list = self.reload_data()
        test_iter_list = []
        for i in range(len(test_loader_list)):
            test_iter_list.append(iter(test_loader_list[i]))

        if self.use_ema:
            self.apply_ema()

        self.net.eval()
        if self.config.args.sample_to_eval:
            for i in range(len(test_loader_list)):
                sample_path = os.path.join(self.config.result.sample_to_eval_path,
                                           self.config.data.dataset_name_list[i])
                if self.config.training.use_DDP:
                    self.sample_to_eval(self.net.module, test_iter_list[i], sample_path)
                else:
                    self.sample_to_eval(self.net, test_iter_list[i], sample_path)
        else:
            for i in range(len(test_iter_list)):
                test_iter = test_iter_list[i]
                for j in tqdm(range(1), initial=0, dynamic_ncols=True, smoothing=0.01):
                    test_batch = next(test_iter)
                    sample_path = os.path.join(self.config.result.sample_path,
                                               self.config.data.dataset_name_list[i],
                                               str(j))
                    if self.config.training.use_DDP:
                        self.sample(self.net.module, test_batch, sample_path, stage='test')
                    else:
                        self.sample(self.net, test_batch, sample_path, stage='test')
