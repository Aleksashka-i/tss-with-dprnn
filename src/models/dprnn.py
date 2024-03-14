import torch
from torch import nn

class SingleRNN(nn.Module):
    """
    Single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int.
        hidden_size: int.
        dropout: float, default: 0.
        bidirectional: bool, default: False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.rnn = getattr(nn, rnn_type)(input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=1,
                                         dropout=dropout,
                                         batch_first=True,
                                         bidirectional=bidirectional)
    
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, input):
        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.
        output = input
        rnn_output, _ = self.rnn(output)
        return rnn_output

class DPRNNBlock(nn.Module):
    """Dual-Path RNN Block.

    Args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int.
        hidden_size: int.
        dropout: float, default: 0.
        bidirectional: bool, default: False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=True):
        super().__init__()
        self.intra_RNN = SingleRNN(
            rnn_type,
            input_size,
            hidden_size,
            dropout=dropout,
            bidirectional=True, # always bi-directional
        )
        self.intra_linear = nn.Linear(self.intra_RNN.output_size, input_size)
        self.intra_norm = nn.GroupNorm(1, input_size)

        self.inter_RNN = SingleRNN(
            rnn_type,
            input_size,
            hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.inter_linear = nn.Linear(self.inter_RNN.output_size, input_size)
        self.inter_norm = nn.GroupNorm(1, input_size)

    def forward(self, input):
        """Input shape : [batch, N (feature dimension), chunk_size, num_chunks]"""
        B, N, K, S = input.size()
        output = input
        # Intra-chunk processing
        input = input.transpose(1, -1).reshape(B * S, K, N)
        input = self.intra_RNN(input)
        input = self.intra_linear(input)
        input = input.reshape(B, S, K, N).transpose(1, -1)
        input = self.intra_norm(input)
        output = output + input # residual connection
        # Inter-chunk processing
        input = output.transpose(1, 2).transpose(2, -1).reshape(B * K, S, N)
        input = self.inter_RNN(input)
        input = self.inter_linear(input)
        input = input.reshape(B, K, S, N).transpose(1, -1).transpose(2, -1).contiguous()
        input = self.inter_norm(input)
        return output + input