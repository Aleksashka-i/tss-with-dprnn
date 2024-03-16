import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import soundfile as sf
import random as random

class BaseDataset(Dataset):
    """Base dataset (Libri2Mix) class for DPRNN-TasNet.

    Args:
        csv_path (str): The path to the metadata file.
        sample_rate (float): The sample rate of the sources and mixtures.
        segment (int, optional): The desired sources and mixtures length in s.
        return_id (bool): If True, returns mixture ID. Default is False.
    """
    def __init__(self, csv_path, sample_rate, segment=3, return_id=False):
        self.csv_path = csv_path
        self.sample_rate = sample_rate
        self.segment = segment
        self.return_id = return_id
        self.n_src = 2
        # Open csv file
        self.df = pd.read_csv(self.csv_path)
         # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        mixture_path = row["mixture_path"]
        sources_list = []
        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # Read sources
        for i in range(self.n_src):
            source_path = row[f"source_{i + 1}_path"]
            s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
            sources_list.append(s)
        # Read the mixture
        mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        # Convert to torch tensor
        mixture = torch.from_numpy(mixture)
        # Stack sources
        sources = np.vstack(sources_list)
        # Convert sources to tensor
        sources = torch.from_numpy(sources)
        if not self.return_id:
            return mixture, sources
        # e.g 5400-34479-0005_4973-24515-0007.wav => 5400-34479-0005, 4973-24515-0007
        id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
        return mixture, sources, [id1, id2]


def getTrainDataloader(config):
    """Returns a DataLoader object based on the given config (train version).

    Parameters
    ----------
    dict config -- the config
    """
    train_set = BaseDataset(
        csv_path=config["data"]["train_dir"],
        sample_rate=config["data"]["sample_rate"],
        segment=config["data"]["segment"],
    )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        drop_last=True,
    )
    return train_loader

def getEvalDataloader(config):
    """Returns a DataLoader object based on the given config (eval version).

    Parameters
    ----------
    dict config -- the config
    """
    eval_set = BaseDataset(
        csv_path=config["data"]["valid_dir"],
        sample_rate=config["data"]["sample_rate"],
        segment=config["data"]["segment"],
    )
    eval_loader = DataLoader(
        eval_set,
        shuffle=False,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        drop_last=True,
    )
    return eval_loader