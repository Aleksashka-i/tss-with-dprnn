import os
import random
import shutil
import zipfile

import numpy as np
import pandas as pd

import torch
from torch import hub
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import soundfile as sf

MINI_URL = 'https://zenodo.org/record/3871592/files/MiniLibriMix.zip?download=1'

class Librimix(Dataset):
    '''Base dataset (Libri2Mix) class for DPRNN-TasNet.

    Args:
        csv_path (str): The path to the metadata file.
        sample_rate (float): The sample rate of the sources and mixtures.
        segment (int, optional): The desired sources and mixtures length in s.
        return_id (bool): If True, returns mixture ID. Default is False.
    '''
    def __init__(self, csv_path, sample_rate, nrows = None, segment=3, return_id=False):
        self.csv_path = csv_path
        self.sample_rate = sample_rate
        self.segment = segment
        self.return_id = return_id
        self.n_src = 2
        # Open csv file
        if nrows is None:
            self.df = pd.read_csv(self.csv_path)
        else:
            self.df = pd.read_csv(self.csv_path, nrows=nrows)
         # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df['length'] >= self.seg_len]
            print(
                f'Drop {max_len - len(self.df)} utterances from {max_len} '
                f'(shorter than {segment} seconds)',
                flush=True
            )
        else:
            self.seg_len = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        mixture_path = row['mixture_path']
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row['length'] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # Read sources
        for i in range(self.n_src):
            source_path = row[f'source_{i + 1}_path']
            s, _ = sf.read(source_path, dtype='float32', start=start, stop=stop)
            sources_list.append(s)
        # Read the mixture
        mixture, _ = sf.read(mixture_path, dtype='float32', start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if not self.return_id:
            return mixture, sources
        # e.g 5400-34479-0005_4973-24515-0007.wav => 5400-34479-0005, 4973-24515-0007
        id1, id2 = mixture_path.split('/')[-1].split('.')[0].split('_')
        return mixture, sources, [id1, id2]

    @classmethod
    def loaders_from_mini(cls, batch_size=4, nrows=None, **kwargs):
        '''Downloads MiniLibriMix and returns train and validation DataLoader.
        '''
        train_set, val_set = cls.mini_from_download(nrows=nrows, **kwargs)
        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)
        return train_loader, val_loader

    @classmethod
    def mini_from_download(cls, nrows=None, **kwargs):
        '''Downloads MiniLibriMix and returns train and validation Dataset.
        '''
        assert 'csv_dir' not in kwargs, 'Cannot specify csv_dir when downloading.'
        assert kwargs.get('task', 'sep_clean') in [
            'sep_clean',
            'sep_noisy',
        ], 'Only clean and noisy separation are supported in MiniLibriMix.'
        assert (
            kwargs.get('sample_rate', 8000) == 8000
        ), 'Only 8kHz sample rate is supported in MiniLibriMix.'
        # Download LibriMix in current directory
        meta_path = cls.mini_download()
        print(meta_path, flush=True)
        # Create dataset instances
        train_set = cls(os.path.join(meta_path, 'train/mixture_train_mix_clean.csv'),
                        sample_rate=8000, nrows=nrows)
        val_set = cls(os.path.join(meta_path, 'val/mixture_val_mix_clean.csv'),
                      sample_rate=8000, nrows=nrows)
        return train_set, val_set

    @staticmethod
    def mini_download():
        '''Downloads MiniLibriMix from Zenodo in current directory
        Returns:
            The path to the metadata directory.
        '''
        mini_dir = './MiniLibriMix/'
        os.makedirs(mini_dir, exist_ok=True)
        # Download zip (or cached)
        zip_path = mini_dir + 'MiniLibriMix.zip'
        if not os.path.isfile(zip_path):
            hub.download_url_to_file(MINI_URL, zip_path)
        # Unzip zip
        cond = all(os.path.isdir('MiniLibriMix/' + f) for f in ['train', 'val', 'metadata'])
        if not cond:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('./')  # Will unzip in MiniLibriMix
        # Reorder metadata
        src = 'MiniLibriMix/metadata/'
        for mode in ['train', 'val']:
            dst = f'MiniLibriMix/metadata/{mode}/'
            os.makedirs(dst, exist_ok=True)
            [
                shutil.copyfile(src + f, dst + f)
                for f in os.listdir(src)
                if mode in f and os.path.isfile(src + f)
            ]
        return './MiniLibriMix/metadata'

def get_train_dataloader(config):
    '''Returns a DataLoader object based on the given config (train version).

    Parameters
    ----------
    dict config -- the config
    '''
    train_set = Librimix(
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

def get_eval_dataloader(config):
    '''Returns a DataLoader object based on the given config (eval version).

    Parameters
    ----------
    dict config -- the config
    '''
    eval_set = Librimix(
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
