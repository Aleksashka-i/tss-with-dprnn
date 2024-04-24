import sys
import time

from asteroid.metrics import get_metrics

import torch

sys.path.append('../../')
from src.reporters.reporter import Reporter
from src.trainers.trainer import Trainer

class TrainerSpe(Trainer):
    ''' Trainer class. '''
    def __init__(self, model, logger, eval_mixtures, reporter: Reporter, config):
        super().__init__(model, logger, eval_mixtures, reporter, config)
        self.ce_gamma = config['ce_gamma']

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
        for step, (mix, target, reference, spk_idx) in enumerate(dataloader):
            mix_device = mix.to(self.device)
            target_device = target.to(self.device)
            reference_device = reference.to(self.device)
            spk_idx = spk_idx.to(self.device)
            self.optimizer.zero_grad()
            ref_len = torch.tensor(reference_device.shape[1], dtype=torch.float32, device=self.device)
            est, aux = self.model(mix_device, reference_device, ref_len)

            l = self.loss_module(est.unsqueeze(1), target_device.unsqueeze(1))
            ce = torch.nn.CrossEntropyLoss()
            ce_loss = ce(aux, spk_idx)

            epoch_loss = l + self.ce_gamma * ce_loss
            total_loss += epoch_loss.item()
            epoch_loss.backward()

            if self.is_metrics is True:
                mix = mix.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                est = est.detach().cpu().numpy()
                for mix_, target_, est_ in zip(mix, target, est):
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
                self.logger.info('l: {}, ce: {}'.format(l, ce_loss))
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
            for step, (mix, target, reference, _) in enumerate(dataloader):
                mix = mix.to(self.device)
                target = target.to(self.device)
                reference = reference.to(self.device)

                ref_len = torch.tensor(reference.shape[1], dtype=torch.float32, device=self.device)
                est, _ = self.model(mix, reference, ref_len)

                l = self.loss_module(est.unsqueeze(1), target.unsqueeze(1))
                epoch_loss = l
                total_loss += epoch_loss.item()

                if self.is_metrics is True:
                    mix = mix.cpu().numpy()
                    target = target.cpu().numpy()
                    est = est.cpu().numpy()
                    for mix_, target_, est_ in zip(mix, target, est):
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
                reference = mix_id['reference'].unsqueeze(0)

                mix = mix.to(self.device)
                reference = reference.to(self.device)
                ref_len = torch.tensor(reference.shape[1], dtype=torch.float32, device=self.device)
                est, _ = self.model(mix, reference, ref_len)

                self.eval_mixtures[id]['estimated'] = est.squeeze(0)
            logs={}
            logs['step'] = self.cur_epoch
            logs['mixtures'] = self.eval_mixtures
            self.reporter.add_and_report(logs=logs, mode='inference_spe')