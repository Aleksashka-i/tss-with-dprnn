from torch import nn
from torch.nn.functional import fold, unfold
from src.models.encoder_decoder import Encoder, Decoder

class SingleRNN(nn.Module):
    ''' Single RNN layer.

    Args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int.
        hidden_size: int.
        dropout: float, default: 0.
        bidirectional: bool, default: False.
    '''

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
    ''' Dual-Path RNN Block.

    Args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int.
        hidden_size: int.
        dropout: float, default: 0.
        bidirectional: bool, default: False.
    '''

    def __init__(self, rnn_type, feature_size, hidden_size, dropout=0, bidirectional=True):
        super().__init__()

        self.intra_RNN = SingleRNN(
            rnn_type,
            feature_size,
            hidden_size,
            dropout=dropout,
            bidirectional=True, # always bi-directional
        )
        self.intra_linear = nn.Linear(self.intra_RNN.output_size(), feature_size)
        self.intra_norm = nn.GroupNorm(1, feature_size)

        self.inter_RNN = SingleRNN(
            rnn_type,
            feature_size,
            hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.inter_linear = nn.Linear(self.inter_RNN.output_size(), feature_size)
        self.inter_norm = nn.GroupNorm(1, feature_size)

    def forward(self, input):
        '''Input shape : [batch, N (feature dimension), chunk_size, num_chunks]'''
        B, N, K, S = input.size()
        output = input
        # Intra-chunk processing
        input = input.transpose(1, -1).reshape(B * S, K, N) # -> [BS, K, N]
        input = self.intra_RNN(input) # -> # [BS, K, N]
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

class DPRNN(nn.Module):
    ''' Dual-Path RNN.

    Args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int.
        feature_size: int, default: 128.
        hidden_size: int, default: 128.
        chunk_length: int, 
        hop_length: int,
        n_repeats: int, default: 6.
        dropout: float, default: 0.
        bidirectional: bool, default: False.
    '''
    def __init__(self, input_size, output_size=None, feature_size=128, hidden_size=128, chunk_length=200,
                 hop_length=None, n_repeats=6, bidirectional=True, rnn_type='LSTM', dropout=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.feature_size = feature_size
        self.chunk_length = chunk_length # length of chunk in segmentation
        self.hop_length = hop_length if hop_length is not None else chunk_length // 2 # length of hop in segmentation

        # bottleneck
        linear_norm = nn.GroupNorm(1, input_size, eps=1e-8)
        start_conv1d = nn.Conv1d(input_size, feature_size, 1, bias=False)
        self.bottleneck = nn.Sequential(linear_norm, start_conv1d)

        # stack dprnn blocks
        dprnn_blocks = []
        for _ in range(n_repeats):
            dprnn_blocks += [
                DPRNNBlock(
                    rnn_type=rnn_type,
                    feature_size=feature_size,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            ]
        self.dprnn_blocks = nn.Sequential(*dprnn_blocks)

        # masks
        self.prelu = nn.PReLU()
        self.conv2d = nn.Conv2d(feature_size, feature_size * 2, kernel_size=1)

        # those are from asteroid, not sure?
        self.out = nn.Sequential(nn.Conv1d(feature_size, feature_size, 1), nn.Tanh())
        self.gate = nn.Sequential(nn.Conv1d(feature_size, feature_size, 1), nn.Sigmoid())

        self.end_conv1x1 = nn.Conv1d(feature_size, self.output_size, 1, bias=False)
        self.activation = nn.ReLU()
    
    def forward(self, input):
        '''
           input: [B, N(input_size), L]
        '''
        B, _, L = input.size()
        output = self.bottleneck(input)  # -> [B, N(feature_size), L]
        output, n_chunks = self._segmentation(output) # ->  # [B, N, K, S]
        output = self.dprnn_blocks(output)
        output = self.prelu(output)
        output = self.conv2d(output) # ->  # [B, 2 * N(feature_size), K, S]
        output = output.reshape(B * 2, self.feature_size, self.chunk_length, n_chunks) # -> [2 * B, N(feature_size), K, S]
        output = self._overlap_add(output, L) # -> [2 * B, N(feature_size), L]
        output = self.out(output) * self.gate(output)
        output = self.end_conv1x1(output) # -> [2 * B, N(output_size), L]
        output = output.reshape(B, 2, self.output_size, L) # -> [B, 2, N(output_size), L]
        output = self.activation(output)
        output = output.transpose(0, 1) # -> [2, B, N(output_size), L]
        return output

    def _segmentation(self, input):
        '''
           Segmentation stage:

           K: length of chunks
           P: hop size
           input: [B, N(input_size), L]
           output: [B, N(input_size), K, S]
        '''
        B, _, _ = input.size()
        output = input
        output = unfold(
            output.unsqueeze(-1),
            kernel_size=(self.chunk_length, 1),
            padding=(self.chunk_length, 0),
            stride=(self.hop_length, 1),
        )
        n_chunks = output.shape[-1]
        output = output.reshape(B, self.input_size, self.chunk_length, n_chunks)
        return output, n_chunks

    
    def _overlap_add(self, input, L):
        '''
           Overlap-add stage:

           input: [2 * B, N(feature_size), K, S]
           gap: padding length
           output: [2 * B, N(feature_size), L]
        '''
        batchx2, _, _, n_chunks = input.size()
        output = input
        to_unfold = self.feature_size * self.chunk_length
        output = fold(
            output.reshape(batchx2, to_unfold, n_chunks),
            (L, 1),
            kernel_size=(self.chunk_length, 1),
            padding=(self.chunk_length, 0),
            stride=(self.hop_length, 1),
        )
        output = output.reshape(batchx2, self.feature_size, -1)
        return output

class DPRNNTasNet(nn.Module):
    '''
       DPRNN-TasNet

       Args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int.
        feature_size: int, default: 128.
        hidden_size: int, default: 128.
        kernel_size: int, default: 2(?).
        chunk_length: int, 
        hop_length: int,
        n_repeats: int, default: 6.
        dropout: float, default: 0.
        bidirectional: bool, default: False.
        stride: int.
        sample_rate: float.
    '''
    def __init__(self, input_size, output_size=None, feature_size=128, hidden_size=128, chunk_length=200,
                 kernel_size=2, hop_length=None, n_repeats=6, bidirectional=True, rnn_type='LSTM',
                 dropout=0, stride=None):
        super().__init__()
        self.output_size = output_size if output_size is not None else input_size
        self.encoder = Encoder(
            kernel_size, 
            input_size
        )
        self.separation = DPRNN(
            input_size,
            output_size,
            feature_size, 
            hidden_size,
            chunk_length,
            hop_length,
            n_repeats,
            bidirectional,
            rnn_type,
            dropout,
        )
        self.decoder = Decoder(
            in_channels=output_size,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
    
    def forward(self, input):
        '''
           input: [B, L]
        '''
        encoders = self.encoder(input) # -> [B, N, L]
        masks = self.separation(encoders) # [2, B, N, L]
        output = [masks[i] * encoders for i in range(2)]
        mixtures = [self.decoder(output[i]) for i in range(2)]
        return mixtures