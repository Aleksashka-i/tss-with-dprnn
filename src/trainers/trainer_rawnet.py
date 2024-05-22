import time

import torch
import torchaudio.transforms as T

from src.reporters.reporter import Reporter
from src.trainers.trainer import Trainer

class TrainerRawNet(Trainer):
    ''' Trainer (spe) class. '''
    def __init__(self, model, logger, eval_mixtures, reporter: Reporter, config):
        super().__init__(model, logger, eval_mixtures, reporter, config)
        self.ce_gamma = config['ce_gamma']
        sample_rate = 8000
        resample_rate = 16000
        self.resampler = T.Resample(sample_rate, resample_rate, dtype=torch.float32)

    def train(self, dataloader):
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
            reference = self.resampler(reference)
            
            mix_device = mix.to(self.device)
            target_device = target.to(self.device)
            reference_device = reference.to(self.device)
            spk_idx = spk_idx.to(self.device)
            self.optimizer.zero_grad()
            est, aux = self.model(mix_device, reference_device)

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
                metric_dict = self._get_metric(mix, target, est, metric_dict)

            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

            self.optimizer.step()

            if step % self.print_freq == 0:
                self.logger.info('l: {}, ce: {}'.format(l, ce_loss))
                self._log_step(step, total_loss)
        end_time = time.time()

        total_loss = self._log_epoch(
            total_loss,
            num_steps,
            metric_dict,
            metric_cnt,
            start_time,
            end_time,
            'train',
        )
        return total_loss

    def eval(self, dataloader):
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
                reference = self.resampler(reference)

                mix = mix.to(self.device)
                target = target.to(self.device)
                reference = reference.to(self.device)

                est, _ = self.model(mix, reference)

                l = self.loss_module(est.unsqueeze(1), target.unsqueeze(1))
                epoch_loss = l
                total_loss += epoch_loss.item()

                if self.is_metrics is True:
                    mix = mix.cpu().numpy()
                    target = target.cpu().numpy()
                    est = est.cpu().numpy()
                    metric_dict = self._get_metric(mix, target, est, metric_dict)

                if step % self.print_freq == 0:
                    self._log_step(step, total_loss)

        end_time = time.time()

        total_loss = self._log_epoch(
            total_loss,
            num_steps,
            metric_dict,
            metric_cnt,
            start_time,
            end_time,
            'eval',
        )
        return total_loss

    def _mixtures_inference(self):
        with torch.no_grad():
            for id in self.eval_mixtures:
                mix_id = self.eval_mixtures[id]

                mix = mix_id['mix'].unsqueeze(0)
                reference = mix_id['reference'].unsqueeze(0)
                reference = self.resampler(reference)

                mix = mix.to(self.device)
                reference = reference.to(self.device)
                est, _ = self.model(mix, reference)

                self.eval_mixtures[id]['estimated'] = est.squeeze(0)
            logs={}
            logs['step'] = self.cur_epoch
            logs['mixtures'] = self.eval_mixtures
            self.reporter.add_and_report(logs=logs, mode='inference_spe')
