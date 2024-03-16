import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """ Conv-TasNet Encoder.

    Args:
        kernel_size: int.
        out_channels: int.
    """
    def __init__(self, kernel_size, output_size):
        super().__init__()
        self.conv1d = nn.Conv1d(1, output_size, kernel_size, stride=kernel_size // 2, bias=False)

    def forward(self, input):
        """
        Input:
            input: [B, T], B is batch size, T is times
        Returns:
            output: [B, C, T_out]
            - T_out is the number of time steps
            - C is out_channels
        """
        output = input
        output = torch.unsqueeze(output, 1)  # -> [B, 1, T]
        output = F.relu(self.conv1d(output))  # -> [B, C, T_out]
        return output

class Decoder(nn.ConvTranspose1d):
    """ Conv-TasNet Decoder. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        raise NotImplementedError