import sys
import os
import time
from collections import deque

from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
from asteroid.metrics import get_metrics

import hydra
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

sys.path.append('../../')
from src.reporters.reporter import Reporter

class Trainer:
    ''' Trainer class. '''
    def __init__(self, model, logger, eval_mixtures, reporter: Reporter, config):
        self.logger = logger
        self.reporter = reporter
        self.cur_epoch = config['cur_epoch']
        self.print_freq = config['print_freq']
        self.eval_mixtures = eval_mixtures # displayed in audio inference table in wandb
        self.sample_rate = config['sample_rate']
        self.metrics = ['si_sdr', 'pesq', 'stoi']
        self.is_metrics = config['is_metrics']

        # choosing the device
        if torch.cuda.is_available():
            self.logger.info('CUDA is available, using GPU for computations.')
        else:
            self.logger.info('CUDA is unavailable, using CPU for computations.')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # model
        self.model = model.to(self.device)

        # loss module
        self.loss_module = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

        # init training
        self.trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = hydra.utils.instantiate(config['optimizer'], params=self.trainable_params)
        self.decay_rate = config['lr_scheduler']['decay_rate']
        if self.decay_rate is not None:
            self.logger.info('lr_scheduler is ExponentialLR.')
            self.lr_scheduler = ExponentialLR(
                optimizer=self.optimizer,
                gamma=config['lr_scheduler']['decay_rate']
            )
        else:
            self.logger.info('lr_scheduler is ReduceLROnPlateau.')
            self.lr_scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer,
                factor=config['lr_scheduler']['factor'],
                patience=config['lr_scheduler']['patience'],
            )

        # clip norm
        if config['clip_norm']:
            self.clip_norm = config['clip_norm']
            self.logger.info('Gradient clipping by {}.'.format(self.clip_norm))
        else:
            self.clip_norm = 0

        #checkpoint handling
        checkpoint_path = config['checkpoint_path']
        if checkpoint_path is not None:
            self.logger.info(f'Continue training from checkpoint: {checkpoint_path}.')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            try:
                self.model.load_state_dict(checkpoint['model'])
            except Exception as e:
                self.logger.info(e)
                self.logger.info('WARNING! load_state_dict_failed,'
                                 'expected load_state_dict in the checkpoint init.')
            self.model = self.model.to(self.device)
        else:
            self.logger.info('Starting new training run.')

        #checkpoint configuration
        self.checkpoint_queue = deque(maxlen=config.n_checkpoints)
        self.new_checkpoints_path = config['new_checkpoints_path']
        self.logger.info(self.new_checkpoints_path)

    def train(self, dataloader):
        ''' Train stage. '''
        self.logger.info('Set train mode...')
        self.model.train()
        num_steps = len(dataloader)
        total_loss = 0.0
        if self.is_metrics is True:
            metric_dict = {metric: 0.0 for metric in self.metrics}
        else:
            metric_dict = None
        metric_cnt = 0

        start_time = time.time()
        for step, (mix, ref) in enumerate(dataloader):
            mix_device = mix.to(self.device)
            ref_device = ref.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(mix_device)

            l, reordered_sources = self.loss_module(out, ref_device, return_est=True)
            epoch_loss = l
            total_loss += epoch_loss.item()
            epoch_loss.backward()

            if self.is_metrics is True:
                mix = mix.detach().cpu().numpy()
                ref = ref.detach().cpu().numpy()
                reordered_sources = reordered_sources.detach().cpu().numpy()
                for mix_, target_, est_ in zip(mix, ref, reordered_sources):
                    metric_cnt += 1
                    mix_ = mix_.reshape((1, ) + mix_.shape)
                    cur_metrics_dict = get_metrics(
                        mix_,
                        target_,
                        est_,
                        sample_rate=self.sample_rate,
                        metrics_list=self.metrics
                    )
                    metric_dict = {metric: metric_dict[metric] + cur_metrics_dict[metric]
                               for metric in self.metrics}

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

            self.optimizer.step()

            if step % self.print_freq == 0:
                message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>.'.format(
                    self.cur_epoch,
                    step,
                    self.optimizer.param_groups[0]['lr'],
                    -total_loss / (step + 1)
                )
                self.logger.info(message)
        end_time = time.time()

        total_loss = total_loss / num_steps
        if self.is_metrics is True:
            metric_dict = {metric: metric_dict[metric] / metric_cnt for metric in self.metrics}

        logs={}
        logs['step'] = self.cur_epoch
        logs['loss'] = -total_loss
        logs['metrics'] = metric_dict
        self.reporter.add_and_report(logs=logs, mode='train')

        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min>.'.format(
            self.cur_epoch,
            num_steps,
            self.optimizer.param_groups[0]['lr'],
            -total_loss,
            (end_time - start_time) / 60
        )
        self.logger.info(message)
        return total_loss

    def eval(self, dataloader):
        ''' Eval stage. '''
        self.logger.info('Set eval mode...')
        self.model.eval()
        num_steps = len(dataloader)
        total_loss = 0.0
        if self.is_metrics is True:
            metric_dict = {metric: 0.0 for metric in self.metrics}
        else:
            metric_dict = None
        metric_cnt = 0

        start_time = time.time()
        with torch.no_grad():
            for step, (mix, ref) in enumerate(dataloader):
                mix = mix.to(self.device)
                ref = ref.to(self.device)

                out = self.model(mix)

                l, reordered_sources = self.loss_module(out, ref, return_est=True)
                epoch_loss = l
                total_loss += epoch_loss.item()

                if self.is_metrics is True:
                    mix = mix.cpu().numpy()
                    ref = ref.cpu().numpy()
                    reordered_sources = reordered_sources.cpu().numpy()
                    for mix_, target_, est_ in zip(mix, ref, reordered_sources):
                        metric_cnt += 1
                        mix_ = mix_.reshape((1, ) + mix_.shape)
                        cur_metrics_dict = get_metrics(
                            mix_,
                            target_,
                            est_,
                            sample_rate=self.sample_rate,
                            metrics_list=self.metrics
                        )
                        metric_dict = {metric: metric_dict[metric] + cur_metrics_dict[metric]
                                   for metric in self.metrics}

                if step % self.print_freq == 0:
                    message = '<epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}>.'.format(
                        self.cur_epoch,
                        step,
                        self.optimizer.param_groups[0]['lr'],
                        -total_loss / (step + 1)
                    )
                    self.logger.info(message)

        end_time = time.time()

        total_loss = total_loss / num_steps
        if self.is_metrics is True:
            metric_dict = {metric: metric_dict[metric] / metric_cnt for metric in self.metrics}

        logs={}
        logs['step'] = self.cur_epoch
        logs['loss'] = -total_loss
        logs['metrics'] = metric_dict
        self.reporter.add_and_report(logs=logs, mode='eval')

        message = 'Finished *** <epoch:{:d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min>.'.format(
            self.cur_epoch,
            num_steps,
            self.optimizer.param_groups[0]['lr'],
            -total_loss,
            (end_time - start_time) / 60
        )
        self.logger.info(message)
        return total_loss

    def run(self, train_loader, eval_loader, n_epochs, early_stop):
        ''' Run. '''
        best_loss = 100500

        train_losses = []
        eval_losses = []
        no_improve_cnt = 0

        while self.cur_epoch < n_epochs:
            self.logger.info('Initiating epoch '+ str(self.cur_epoch) + '.')
            self.cur_epoch += 1

            train_loss = self.train(train_loader)
            eval_loss = self.eval(eval_loader)

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

            if self.decay_rate is None:
                self.lr_scheduler.step(eval_loss)
            else:
                self.lr_scheduler.step()

            if eval_loss >= best_loss:
                no_improve_cnt += 1
                self.logger.info('No improvement, Best Loss: {:.4f}.'.format(-best_loss))
            else:
                best_loss = eval_loss
                no_improve_cnt = 0
                self.save_checkpoint(best=True)
                self.logger.info('Epoch: {:d}, Now Best Loss Change: {:.4f}.'
                                 .format(self.cur_epoch, -best_loss))

                self.mixtures_inference()

            if no_improve_cnt == early_stop:
                self.logger.info('Stop training cause no impr for {:d} epochs'
                                 .format(no_improve_cnt))
                break

        self.save_checkpoint(best=False)
        self.logger.info('Training for {:d}/{:d} epoches done!'.format(self.cur_epoch, n_epochs))

    def mixtures_inference(self):
        ''' Audio inference for eval_mixtures '''
        with torch.no_grad():
            for id in self.eval_mixtures:
                mix_id = self.eval_mixtures[id]
                mix = mix_id['mix'].unsqueeze(0)
                mix = mix.to(self.device)
                out = self.model(mix)
                sources = torch.stack([mix_id['s1_target'].unsqueeze(0),
                                       mix_id['s2_target'].unsqueeze(0)])
                sources = sources.transpose(0, 1)
                sources = sources.to(self.device)
                _, reordered_sources = self.loss_module(out, sources, return_est=True)
                reordered_sources = reordered_sources.squeeze(0)
                self.eval_mixtures[id]['s1_estimated'] = reordered_sources[0]
                self.eval_mixtures[id]['s2_estimated'] = reordered_sources[1]
            logs={}
            logs['step'] = self.cur_epoch
            logs['mixtures'] = self.eval_mixtures
            self.reporter.add_and_report(logs=logs, mode='inference')

    def process_checkpoint(self, path):
        ''' Directory maintaining. '''
        if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
            removed_checkpoint = self.checkpoint_queue[0]
            os.remove(removed_checkpoint)
        self.checkpoint_queue.append(path)

    def save_checkpoint(self, best=False):
        ''' Saves the checkpoint. '''
        cpt = {
            'epoch': self.cur_epoch,
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.state_dict()
        }
        path_to_save = os.path.join(self.new_checkpoints_path,
                                    '{0}_{1}.pt'.format(str(self.cur_epoch), 'best' if best else 'last'))
        torch.save(cpt, path_to_save)
        self.process_checkpoint(path_to_save)
