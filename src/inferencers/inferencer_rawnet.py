import sys
import time
import pandas as pd

from asteroid.metrics import get_metrics

import torch

from src.inferencers.inferencer import Inferencer

import torchaudio.transforms as T

sys.path.append('../../')
from src.reporters.reporter import Reporter

class InferencerRawNet(Inferencer):
    ''' Inferencer (Spe) class. '''
    def __init__(self, model, logger, config, reporter: Reporter):
        super().__init__(model, logger, config)
        self.reporter = reporter
        sample_rate = 8000
        resample_rate = 16000
        self.resampler = T.Resample(sample_rate, resample_rate, dtype=torch.float32)

    def run(self, test_set):
        series_list = []

        start_time = time.time()
        self.model.eval()
        torch.no_grad().__enter__()
        for idx in range(len(test_set)):
            self.logger.info('idx: {}'.format(idx))
            mix, target, reference, _ = test_set[idx]
            mix = mix.to(self.device)

            reference = self.resampler(reference)

            reference = reference.to(self.device)

            est, _ = self.model(mix.unsqueeze(0), reference.unsqueeze(0))

            mix_np = mix.cpu().data.numpy()
            target_np = target.cpu().data.numpy()
            est_np = est.cpu().data.numpy()
            metrics = get_metrics(
                mix_np,
                target_np,
                est_np,
                sample_rate=self.sample_rate,
                metrics_list=self.metrics,
            )
            self.add_result(idx, mix, target, est.squeeze(0), reference, metrics)
            series_list.append(pd.Series(metrics))
        end_time = time.time()

        message = 'Finished *** <Total time:{:.3f} min>.'.format(
            (end_time - start_time) / 60
        )
        self.logger.info(message)
        final_results = self._save_result(series_list)

        self.reporter.add_and_report(logs=pd.DataFrame([final_results]), mode='test_final')
    
    def add_result(self, idx, mix, target, estimated, reference, cur_metrics):
        logs = {}
        logs['id'] = idx
        logs['mix'] = mix
        logs['target'] = target
        logs['estimated'] = estimated
        logs['reference'] = reference
        for metric_name in self.metrics:
            input_metric_name = 'input_' + metric_name
            ldf = cur_metrics[metric_name] - cur_metrics[input_metric_name]
            logs[metric_name] = cur_metrics[metric_name]
            logs[metric_name + '_imp'] = ldf
        self.reporter.add_and_report(logs=logs, mode='test')
