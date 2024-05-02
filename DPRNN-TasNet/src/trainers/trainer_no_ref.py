import time

import torch

from src.trainers.trainer import Trainer

class TrainerNoRef(Trainer):
    ''' Trainer (with no reference) class. '''
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
        for step, (mix, target) in enumerate(dataloader):
            mix_device = mix.to(self.device)
            target = target[:, 0:1, :]
            target_device = target.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(mix_device)
            est = out[:, 0:1, :]

            l = self.loss_module(est, target_device)
            epoch_loss = l
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
            for step, (mix, target) in enumerate(dataloader):
                mix = mix.to(self.device)
                target = target[:, 0:1, :]
                target = target.to(self.device)

                out = self.model(mix)
                est = out[:, 0:1, :]

                l = self.loss_module(est, target)
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

                mix = mix.to(self.device)
                est = self.model(mix)
                est = est[:, 0:1, :]

                self.eval_mixtures[id]['estimated'] = est.squeeze(0).squeeze(0)
            logs={}
            logs['step'] = self.cur_epoch
            logs['mixtures'] = self.eval_mixtures
            self.reporter.add_and_report(logs=logs, mode='inference_no_target')
