import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from itertools import permutations

def Loss(ests, refs, device):
    def si_sdr_loss(permute):
        si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=True).to(device)
        return sum([si_sdr(ests[s], refs[t]) for s, t in enumerate(permute)]) / len(permute)

    B = refs[0].size(0)
    sisdr_mat = torch.stack([si_sdr_loss(p) for p in permutations(range(2))])
    max_per_utt, _ = torch.max(sisdr_mat, dim=0)
    return -torch.sum(max_per_utt) / B