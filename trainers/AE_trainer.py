from torch import nn
import torch
import numpy as np
import os
import time
from torchvision.utils import save_image
import pickle

from dataloader.dataloader import get_dataloaders
from models.Autoencoder import AE
from utils.logging import Logger
from enums.RegType import RegType
from enums.AutoencoderType import AEType
from losses.MSE_contractive_reg import MSE_contractive
from configs.knn_lof_config import cfg as knn_lof_cfg
from utils.KNN import KNN
from utils.LOF import LOF


class Trainer(object):
    def __init__(self, cfg):
        """
        Class for initializing and performing training procedure.
        :param cfg: train config
        """
        self.cfg = cfg
        self.dl_train, self.dl_test = get_dataloaders()
        self.model = self.get_model()
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.logger = Logger(self.cfg)
        self.KNN = KNN(knn_lof_cfg, self.cfg)
        self.LOF = LOF(knn_lof_cfg, self.cfg)

    def get_model(self):
        model = AE(self.cfg.model_cfg)
        return model.cuda()

    def get_criterion(self):
        """
        Gets criterion.
        :return: criterion
        """
        if self.cfg.model_cfg.reg_type == RegType.contractive:
            criterion = MSE_contractive(self.cfg.reg_lambda)
        else:
            criterion = nn.MSELoss()
        return criterion

    def get_optimizer(self):
        """
        Gets optimizer.
        :return: optimizer
        """
        optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.cfg.lr,
                                       'weight_decay': self.cfg.weight_decay}])
        return optimizer

    def restore_model(self):
        """
        Restores saved model.
        """
        if self.cfg.load_saved_model:
            print(f'Trying to load checkpoint from epoch {self.cfg.epoch_to_load}...')
            try:
                checkpoint = torch.load(self.cfg.checkpoints_dir + f'/checkpoint_{self.cfg.epoch_to_load}.pth')
                load_state_dict = checkpoint['model']
                self.model.load_state_dict(load_state_dict)
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step'] + 1
                self.optimizer.load_state_dict(checkpoint['opt'])
                print(f'Loaded checkpoint from epoch {self.cfg.epoch_to_load}.')
            except FileNotFoundError:
                print('Checkpoint not found')

    def save_model(self):
        """
        Saves model.
        """
        if self.cfg.save_model and self.epoch % self.cfg.epochs_saving_freq == 0:
            print('Saving current model...')
            state = {
                'model': self.model.state_dict(),
                'epoch': self.epoch,
                'global_step': self.global_step,
                'opt': self.optimizer.state_dict()
            }
            if not os.path.exists(self.cfg.checkpoints_dir):
                os.makedirs(self.cfg.checkpoints_dir)

            path_to_save = os.path.join(self.cfg.checkpoints_dir, f'checkpoint_{self.epoch}.pth')
            torch.save(state, path_to_save)
            print(f'Saved model to {path_to_save}.')

    def save_vectors(self, dl):
        """
        Saves h output for every image in given dataloader.
        """
        if not os.path.exists(self.cfg.vectors_path):
            os.makedirs(self.cfg.vectors_path)

        outs = []
        self.model.eval()
        with torch.no_grad():
            print(f'Saving vectors...')
            dl_len = len(dl)
            for i, batch in enumerate(dl):
                images = batch[0].cuda()

                if i % 50 == 0:
                    print(f'iter: {i}/{dl_len}')

                out, _ = self.model(images)
                out = out.cpu().numpy().reshape(self.cfg.batch_size, -1)
                outs.extend(out)

            with open(self.cfg.vectors_path + f'test_set_vectors.pickle', 'wb') as f:
                pickle.dump(np.asarray(outs), f)

    def evaluate(self, dl, set_type):
        """
        Evaluates model performance. Calculates and logs model accuracy on given data set.
        :param dl: train or test dataloader
        :param set_type: 'train' or 'test' data type
        """
        if not os.path.exists(self.cfg.eval_plots_dir + f'eval_{set_type}'):
            os.makedirs(self.cfg.eval_plots_dir + f'eval_{set_type}')
        all_predictions, all_labels, losses = [], [], []

        self.model.eval()
        with torch.no_grad():
            print(f'Evaluating on {set_type} data...')
            eval_start_time = time.time()

            dl_len = len(dl)
            for i, batch in enumerate(dl):
                images = batch[0].cuda()

                if i % 50 == 0:
                    print(f'iter: {i}/{dl_len}')

                out, _ = self.model(images)
                loss = self.criterion(out, images)
                losses.append(loss.item())

            save_image(torch.cat([images[:8], out[:8]]), self.cfg.eval_plots_dir +
                       f'eval_{set_type}/predictions_epoch_{self.epoch}.png')

            mean_loss = np.mean(losses)
            print(f'Loss on {set_type} data: {mean_loss}')

            self.logger.log_metrics(names=[f'eval/{set_type}/loss'], metrics=[mean_loss], step=self.epoch)
            print(f'Evaluating time: {round((time.time() - eval_start_time) / 60, 3)} min')
        self.model.train()

    def make_training_step(self, batch):
        """
        Makes single training step.
        :param batch: current batch containing input vector and it`s label
        :return: loss on current batch
        """
        images = batch[0].cuda()

        if self.cfg.model_cfg.reg_type == RegType.contractive:
            images.requires_grad_(True)
            images.retain_grad()

        out, h = self.model(images)

        if self.cfg.model_cfg.reg_type == RegType.contractive:
            loss = self.criterion(out, images, h)
            images.requires_grad_(False)
        else:
            loss = self.criterion(out, images)

        if self.cfg.model_cfg.autoencoder_type == AEType.overcomplete and self.cfg.model_cfg.reg_type == RegType.sparse:
            reg = self.cfg.reg_lambda * torch.sum(torch.abs(h))
            loss += reg
        assert not torch.isnan(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), images, out

    def train(self):
        """
        Runs training procedure.
        """
        total_training_start_time = time.time()
        self.start_epoch, self.epoch, self.global_step = 0, -1, 0

        # restore model if necessary
        self.restore_model()

        # evaluate on train and test data before training
        if self.cfg.evaluate_before_training:
            if self.cfg.evaluate_on_train_set:
                self.evaluate(self.dl_train, set_type='train')
            self.evaluate(self.dl_test, set_type='test')

        if self.cfg.save_vectors:
            self.save_vectors(self.dl_test)

        if self.cfg.run_knn:
            self.KNN.run()

        if self.cfg.run_lof:
            self.LOF.run()

        if not os.path.exists(self.cfg.eval_plots_dir + 'training'):
            os.makedirs(self.cfg.eval_plots_dir + 'training')

        # start training
        print(f'Starting training...')
        iter_num = len(self.dl_train)
        for epoch in range(self.start_epoch, self.cfg.epochs):
            epoch_start_time = time.time()
            self.epoch = epoch
            print(f'Epoch: {self.epoch}/{self.cfg.epochs}')

            losses = []
            images, out = None, None
            for iter_, batch in enumerate(self.dl_train):

                loss, images, out = self.make_training_step(batch)
                self.logger.log_metrics(names=['train/loss'], metrics=[loss], step=self.global_step)

                losses.append(loss)
                self.global_step += 1

                if iter_ % 50 == 0:
                    mean_loss = np.mean(losses[-50:]) if len(losses) > 50 else np.mean(losses)
                    print(f'iter: {iter_}/{iter_num}, loss: {mean_loss}')

            save_image(torch.cat([images[:8], out[:8]]), self.cfg.eval_plots_dir +
                       f'training/predictions_epoch_{self.epoch}.png')

            self.logger.log_metrics(names=['train/mean_loss_per_epoch'], metrics=[np.mean(losses)], step=self.epoch)

            # save model
            self.save_model()

            # evaluate on train and test data
            if self.cfg.evaluate_on_train_set:
                self.evaluate(self.dl_train, set_type='train')
            self.evaluate(self.dl_test, set_type='test')

            print(f'Epoch total time: {round((time.time() - epoch_start_time) / 60, 3)} min')

        print(f'Training time: {round((time.time() - total_training_start_time) / 60, 3)} min')
