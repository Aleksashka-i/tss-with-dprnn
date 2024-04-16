import os
import time
import json

from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper
from asteroid.metrics import get_metrics

import pandas as pd
import torch

class Inferencer:
    ''' Inferencer class. '''
    def __init__(self, model, logger, config):
        self.logger = logger
        self.sample_rate = config['sample_rate']
        self.metrics = ['si_sdr', 'sdr', 'sir', 'sar', 'stoi', 'pesq']
        self.test_savedir = config['test_savedir']

        # choosing the device
        if torch.cuda.is_available():
            self.logger.info('CUDA is available, using GPU for computations.')
        else:
            self.logger.info('CUDA is unavailable, using CPU for computations.')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # model
        self.model = model.to(self.device)

        # loss module (used for source reordering)
        self.loss_module = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

        #checkpoint handling
        checkpoint_path = config['checkpoint_path']
        if checkpoint_path is not None:
            self.logger.info(f'Testing for pretrained: {checkpoint_path}.')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            try:
                self.model.load_state_dict(checkpoint['model'])
            except Exception as e:
                self.logger.info(e)
                self.logger.info('WARNING! load_state_dict_failed,'
                                 'expected load_state_dict in the checkpoint init.')
            self.model = self.model.to(self.device)
        else:
            logger.info('No pretrained model was provided.')
            raise ValueError

    def run(self, test_set):
        ''' Run. '''
        series_list = []

        start_time = time.time()
        torch.no_grad().__enter__()
        for idx in range(len(test_set)):
            # Forward the network on the mixture.
            mix, sources = test_set[idx]
            mix = mix.to(self.device)
            sources = sources.to(self.device)
            out = self.model(mix.unsqueeze(0))
            sources = sources.unsqueeze(0)
            _, reordered_sources = self.loss_module(out, sources, return_est=True)
            mix_np = mix.squeeze(0).cpu().data.numpy()
            sources_np = sources.squeeze(0).cpu().data.numpy()
            est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
            metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=self.sample_rate,
                metrics_list=self.metrics,
            )
            series_list.append(pd.Series(metrics))
        end_time = time.time()

        message = 'Finished *** <Total time:{:.3f} min>.'.format(
            (end_time - start_time) / 60
        )
        self.logger.info(message)

        all_metrics_df = pd.DataFrame(series_list)
        all_metrics_df.to_csv(os.path.join(self.test_savedir, 'all_metrics.csv'))

        final_results = {}
        for metric_name in self.metrics:
            input_metric_name = 'input_' + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + '_imp'] = ldf.mean()

        self.logger.info('Overall metrics :')
        self.logger.info(final_results)

        with open(os.path.join(self.test_savedir, 'final_metrics.json'), 'w') as f:
            json.dump(final_results, f, indent=0)
