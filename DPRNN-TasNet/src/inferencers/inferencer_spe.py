import time
import pandas as pd

from asteroid.metrics import get_metrics

import torch

from src.inferencers.inferencer import Inferencer

class InferencerSpe(Inferencer):
    ''' Inferencer (spe) class. '''
    def run(self, test_set):
        ''' Run. '''
        series_list = []

        start_time = time.time()
        torch.no_grad().__enter__()
        for idx in range(len(test_set)):
            mix, target, reference, _ = test_set[idx]
            mix = mix.to(self.device)
            reference = reference.to(self.device)

            ref_len = torch.tensor(reference.shape[0], dtype=torch.float32, device=self.device)
            est, _ = self.model(mix.unsqueeze(0), reference.unsqueeze(0), ref_len)

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
            series_list.append(pd.Series(metrics))
        end_time = time.time()

        message = 'Finished *** <Total time:{:.3f} min>.'.format(
            (end_time - start_time) / 60
        )
        self.logger.info(message)
        self.save_result(series_list)
