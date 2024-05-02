import torch
from torch import nn
from src.models.encoder_decoder import Encoder, Decoder
from src.models import norms
from src.models.dprnn import DPRNN

class ResBlock(nn.Module):
    ''' Resnet block for speaker encoder to obtain speaker embedding.

    Ref to:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        and
        https://github.com/Jungjee/RawNet/blob/master/PyTorch/model_RawNet.py
    '''
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.maxpool = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        return self.maxpool(y)

class DPRNNSpe(DPRNN):
    ''' Dual-Path RNN. (Spe)

    Args:
        input_size: int.
        feature_size: int.
        hidden_size: int.
        chunk_length: int.
        hop_length: int.
        n_repeats: int.
        bidirectional: bool.
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        norm_type: string, 'gLN' or 'ln'.
        activation_type: string, 'sigmoid' or 'relu'.
        dropout: float.
        mid_resnet_size: int.
        embeddings_size: int.
        num_spks: int.
        kernel_size: int.
        fusion_type: string, 'cat', 'add', 'mul' or 'film'.
    '''
    def __init__(self, input_size, feature_size=128, hidden_size=128,
                 chunk_length=200, hop_length=None, n_repeats=6,
                 bidirectional=True, rnn_type='LSTM', norm_type='gLN',
                 activation_type='sigmoid', dropout=0, mid_resnet_size=256,
                 embeddings_size=128, num_spks=251, kernel_size=2,
                 fusion_type='cat'):
        super().__init__(
            input_size,
            feature_size,
            hidden_size,
            chunk_length,
            hop_length,
            n_repeats,
            bidirectional,
            rnn_type,
            norm_type,
            activation_type,
            dropout=dropout,
        )
        self.kernel_size = kernel_size
        self.fusion_type = fusion_type

        # fusion
        if fusion_type == 'cat':
            start_conv1d = nn.Conv1d(input_size + embeddings_size, feature_size, 1)
        if fusion_type == 'add' or fusion_type == 'mul':
            self.fusion_linear = nn.Linear(embeddings_size, input_size)
            start_conv1d = nn.Conv1d(input_size, feature_size, 1)
        if fusion_type == 'film':
            self.fusion_linear_1 = nn.Linear(embeddings_size, input_size)
            self.fusion_linear_2 = nn.Linear(embeddings_size, input_size)
            start_conv1d = nn.Conv1d(input_size, feature_size, 1)

        # bottleneck
        if norm_type == 'gLN':
            linear_norm = norms.GlobLN(input_size)
        else:
            linear_norm = nn.GroupNorm(1, input_size)
        self.bottleneck = nn.Sequential(linear_norm, start_conv1d)

        # target
        self.spk_encoder = nn.Sequential(
            nn.GroupNorm(1, input_size),
            nn.Conv1d(input_size, feature_size, 1),
            ResBlock(feature_size, feature_size),
            ResBlock(feature_size, mid_resnet_size),
            ResBlock(mid_resnet_size, mid_resnet_size),
            nn.Conv1d(mid_resnet_size, embeddings_size, 1),
        )
        self.pred_linear = nn.Linear(embeddings_size, num_spks)

    def forward(self, input, aux, aux_len):
        # input: [B, N(input_size), L]
        # aux: [B, N(input_size), L]

        B, _, L = input.size()

        # auxiliary
        aux = self._auxiliary(aux, aux_len)
        # -> [B, N(embeddings_size)]

        # norm
        output = self.bottleneck[0](input)
        # -> [B, N(input_size), L]

        # fusion
        output = self._fusion(aux, output, L)

        # 1x1 cnn
        output = self.bottleneck[1](output)
        # -> [B, N(feature_size), L]

        # dprnn blocks
        output = self._dprnn_process(output, B, L)
        # -> [B, 2, N(input_size), L]

        # auxiliary linear
        aux = self.pred_linear(aux)
        # -> [B, num_spks]

        return output, aux

    def _auxiliary(self, aux, aux_len):
        aux = self.spk_encoder(aux)
        # -> [B, embeddings_size, L // 3]
        aux_T = (aux_len - self.kernel_size) // (self.kernel_size // 2) + 1
        aux_T = ((aux_T // 3) // 3) // 3
        aux = torch.sum(aux, -1) / aux_T.view(-1, 1).float()
        # -> [B, embeddings_size]
        return aux

    def _fusion(self, aux, output, L):
        if self.fusion_type == 'cat':
            output = self._concatenation(aux, output, L)
            # -> [B, N(input_size + embeddings_size), L]
        if self.fusion_type == 'add':
            output = self._addition(aux, output, L, self.fusion_linear)
            # -> [B, N(input_size), L]
        if self.fusion_type == 'mul':
            output = self._multiplication(aux, output, L, self.fusion_linear)
            # -> [B, N(input_size), L]
        if self.fusion_type == 'film':
            output = self._film(aux, output, L)
            # -> [B, N(input_size), L]
        return output

    def _concatenation(self, aux, output, L):
        aux_concat = torch.unsqueeze(aux, -1)
        aux_concat  = aux_concat.repeat(1, 1, L)
         # -> [B, N(embeddings_size), L]
        output = torch.cat([output, aux_concat], 1)
        # -> [B, N(input_size + embeddings_size), L]
        return output

    def _addition(self, aux, output, L, fusion_linear):
        aux_add = fusion_linear(aux)
        # -> [B, N(input_size)]
        aux_add = torch.unsqueeze(aux_add, -1)
        aux_add = aux_add.repeat(1, 1, L)
        # -> [B, N(input_size), L]
        output = output + aux_add
        # -> [B, N(input_size, L]
        return output

    def _multiplication(self, aux, output, L, fusion_linear):
        aux_mul = fusion_linear(aux)
        # -> [B, N(input_size)]
        aux_mul = torch.unsqueeze(aux_mul, -1)
        aux_mul = aux_mul.repeat(1, 1, L)
        # -> [B, N(input_size), L]
        output = output * aux_mul
        # -> [B, N(input_size, L]
        return output

    def _film(self, aux, output, L):
        output = self._multiplication(aux, output, L, self.fusion_linear_1)
        # -> [B, N(input_size, L]
        output = self._addition(aux, output, L, self.fusion_linear_2)
        # -> [B, N(input_size, L]
        return output

    def _dprnn_process(self, output, B, L):
        output, n_chunks = self._segmentation(output)
        # ->  # [B, N, K, S]
        output = self.dprnn_blocks(output)
        output = self.prelu(output)
        output = self.conv2d(output)
        # ->  # [B, 2 * N(feature_size), K, S]
        output = output.reshape(B * 2, self.feature_size, self.chunk_length, n_chunks)
        # -> [2 * B, N(feature_size), K, S]
        output = self._overlap_add(output, L)
        # -> [2 * B, N(feature_size), L]
        output = self.out(output) * self.gate(output)
        output = self.end_conv1x1(output)
        # -> [2 * B, N(input_size), L]
        output = self.activation(output)
        output = output.reshape(B, 2, self.input_size, L)
        # -> [B, 2, N(input_size), L]
        return output

class DPRNNSpeTasNet(nn.Module):
    ''' DPRNN-Spe-TasNet

    Args:
        input_size: int.
        feature_size: int.
        hidden_size: int.
        chunk_length: int.
        kernel_size: int.
        hop_length: int.
        n_repeats: int.
        bidirectional: bool.
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        norm_type: string, 'gLN' or 'ln'.
        activation_type: string, 'sigmoid' or 'relu'.
        dropout: float.
        stride: int.
        mid_resnet_size: int.
        embeddings_size: int.
        num_spks: int.
        fusion_type: string, 'cat', 'add', 'mul' or 'film'.
    '''
    def __init__(self, input_size, feature_size=128, hidden_size=128,
                 chunk_length=200, kernel_size=2, hop_length=None,
                 n_repeats=6, bidirectional=True, rnn_type='LSTM',
                 norm_type='gLN', activation_type='sigmoid', dropout=0,
                 stride=None, mid_resnet_size=256, embeddings_size=128,
                 num_spks=251, fusion_type='cat'):
        super().__init__()
        self.stride = stride if stride is not None else kernel_size // 2
        self.encoder = Encoder(
            kernel_size,
            input_size,
            stride=self.stride,
            bias=False,
        )
        self.separation = DPRNNSpe(
            input_size,
            feature_size,
            hidden_size,
            chunk_length,
            hop_length,
            n_repeats,
            bidirectional,
            rnn_type,
            norm_type,
            activation_type,
            dropout,
            mid_resnet_size,
            embeddings_size,
            num_spks,
            kernel_size,
            fusion_type,
        )
        self.decoder = Decoder(
            in_channels=input_size,
            out_channels=1,
            kernel_size=kernel_size,
            stride=self.stride,
            bias=False,
        )

    def forward(self, input, aux, aux_len):
        # input: [B, L]
        # aux: [B, L]

        encoders = self.encoder(input)
        # -> [B, N, L]
        embeddings = self.encoder(aux)
        # -> [B, N, L]
        masks, aux = self.separation(encoders, embeddings, aux_len)
        # -> [B, 2, N, L], [B, num_spks]
        output = masks * encoders.unsqueeze(1)
        # -> [B, 2, N, L]
        mixture = self.decoder(output[:, 0, :, :])
        # [B, L] (first speaker only)
        return mixture, aux
