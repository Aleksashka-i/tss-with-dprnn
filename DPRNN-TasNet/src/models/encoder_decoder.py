import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    ''' Conv-TasNet Encoder.

    Args:
        kernel_size: int.
        output_size: int.
        stride: int.
        bias: bool.
    '''
    def __init__(self, kernel_size, output_size, stride, bias=False):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=output_size,
            kernel_size=kernel_size,
            stride=stride,
            groups=1,
            bias=bias
        )

    def forward(self, input):
        output = input
        # [B, L]
        output = torch.unsqueeze(output, 1)
        # -> [B, 1, L]
        output = self.conv1d(output)
        # -> [B, C, L_out]
        output = F.relu(output)
        return output

class Decoder(nn.ConvTranspose1d):
    ''' Conv-TasNet Decoder. '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input
        # [B, N, L]
        output = super().forward(output)
        if torch.squeeze(output).dim() == 1:
            output = torch.squeeze(output, dim=1)
        else:
            output = torch.squeeze(output)
        # -> [B, L]
        return output
