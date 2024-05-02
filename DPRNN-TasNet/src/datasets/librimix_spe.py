import random
import pickle as pkl
import pandas as pd

import torch
from torch.utils.data import DataLoader
import soundfile as sf

from src.datasets.librimix import Librimix

MINI_URL = 'https://zenodo.org/record/3871592/files/MiniLibriMix.zip?download=1'

class LibrimixSpe(Librimix):
    ''' Base dataset (Libri2Mix) class for DPRNN-TasNet.

    Args:
        csv_path: str.
        sample_rate: float.
        nrows: int.
        segment: int.
        return_id: bool.
    '''
    def __init__(self, csv_path, sample_rate, nrows = None, segment=3, return_id=False):
        super().__init__(csv_path, sample_rate, nrows=nrows, segment=segment, return_id=return_id)
        self.speakers_mapping = {}
        self._map_speakers()
        self.start_ref = []
        self.stop_ref = []
        self.df['reference'] = self.df.apply(self._choose_target_dummy, axis=1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mixture_path = row['mixture_path']
        target_path = row['source_1_path']
        start = self.start[idx]
        stop = self.stop[idx]
        reference_path = row['reference']
        start_ref = self.start_ref[idx]
        stop_ref = self.stop_ref[idx]

        target, _ = sf.read(target_path, dtype='float32', start=start, stop=stop)
        target = torch.from_numpy(target)
        mixture, _ = sf.read(mixture_path, dtype='float32', start=start, stop=stop)
        mixture = torch.from_numpy(mixture)
        reference, _ = sf.read(reference_path, dtype='float32', start=start_ref, stop=stop_ref)
        reference = torch.from_numpy(reference)

        # e.g 5400-34479-0005_4973-24515-0007.wav => 5400-34479-0005
        id = self._get_first_speaker_id(mixture_path)
        mapped_id = self.speakers_mapping[id.split('-')[0]]
        if not self.return_id:
            return mixture, target, reference, mapped_id
        return mixture, target, reference, mapped_id, id

    def _get_first_speaker_id(self, mixture_path):
        return mixture_path.split('/')[-1].split('.')[0].split('_')[0]

    def _map_speakers(self):
        cnt = 0
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            mixture_path = row['mixture_path']
            speaker_id = self._get_first_speaker_id(mixture_path).split('-')[0]
            if speaker_id not in self.speakers_mapping.keys():
                self.speakers_mapping[speaker_id] = cnt
                cnt += 1

    def _choose_target_dummy(self, row):
        mixture_path = row['mixture_path']
        audio_id = self._get_first_speaker_id(mixture_path)
        speaker_id = audio_id.split('-')[0]

        pattern_1 = rf'/{speaker_id}-'
        pattern_2 = rf'_{speaker_id}-'
        pattern_1_exclude = rf'/{audio_id}_'
        pattern_2_exclude = rf'_{audio_id}.'

        matches_source_1 = self.df['source_1_path'].str.contains(pattern_1)
        matches_source_2 = self.df['source_2_path'].str.contains(pattern_2)
        matches_source_1_exclude = self.df['source_1_path'].str.contains(pattern_1_exclude)
        matches_source_2_exclude = self.df['source_2_path'].str.contains(pattern_2_exclude)

        filtered_source_1 = self.df.loc[matches_source_1 & ~matches_source_1_exclude,
                                   ['source_1_path', 'length']]
        filtered_source_2 = self.df.loc[matches_source_2 & ~matches_source_2_exclude,
                                   ['source_2_path', 'length']]

        combined_filtered_sources = pd.concat([
            filtered_source_1.rename(columns={'source_1_path': 'source_path'}),
            filtered_source_2.rename(columns={'source_2_path': 'source_path'})
        ])

        target_row = combined_filtered_sources.sample(n = 1)

        if self.seg_len is not None:
            start = random.randint(0, target_row['length'].item() - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        self.start_ref += [start]
        self.stop_ref += [stop]

        return target_row['source_path'].item()

    @classmethod
    def loaders_from_mini(cls, batch_size=4, nrows=None, **kwargs):
        raise NotImplementedError

    @classmethod
    def mini_from_download(cls, nrows=None, **kwargs):
        raise NotImplementedError

    @staticmethod
    def mini_download():
        raise NotImplementedError

def get_train_spe_dataloader(config):
    if config['data']['use_generated_train'] is not None:
        with open(config['data']['use_generated_train'], 'rb') as file:
            train_set = pkl.load(file)
    else:
        train_set = LibrimixSpe(
            csv_path=config['data']['train_path'],
            sample_rate=config['sample_rate'],
            nrows=config['data']['nrows_train'],
            segment=config['data']['segment'],
        )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        drop_last=True,
    )
    return train_set, train_loader

def get_eval_spe_dataloader(config):
    if config['data']['use_generated_eval'] is not None:
        with open(config['data']['use_generated_eval'], 'rb') as file:
            eval_set = pkl.load(file)
    else:
        eval_set = LibrimixSpe(
            csv_path=config['data']['valid_path'],
            sample_rate=config['sample_rate'],
            nrows=config['data']['nrows_valid'],
            segment=config['data']['segment'],
        )
    eval_loader = DataLoader(
        eval_set,
        shuffle=False,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        drop_last=True,
    )
    return eval_set, eval_loader
