import torch
from torch import nn
from src.models.encoder_decoder import Encoder, Decoder
from src.models.dprnn_spe import DPRNNSpe
from src.models.RawNet3 import RawNet3
from src.models.RawNetBasicBlock import Bottle2neck
import torch.nn.functional as F

class DPRNNRawNet(DPRNNSpe):
    ''' Dual-Path RNN. (RawNet)

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
        O: int.
        P: int.
        embeddings_size: int.
        num_spks: int.
        kernel_size: int.
        fusion_type: string, 'cat', 'add', 'mul' or 'film'.
    '''
    def __init__(self, input_size, feature_size=128, hidden_size=128,
                 chunk_length=200, hop_length=None, n_repeats=6,
                 bidirectional=True, rnn_type='LSTM', norm_type='gLN',
                 activation_type='sigmoid', dropout=0, O=128, P=256,
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
            dropout,
            O,
            P,
            embeddings_size,
            num_spks,
            kernel_size,
            fusion_type,
        )

        # target
        self.spk_encoder = RawNet3(
            Bottle2neck,
            model_scale=8,
            context=True,
            summed=True,
            encoder_type="ECA",
            nOut=embeddings_size,
            out_bn=False,
            sinc_stride=10,
            log_sinc=True,
            norm_sinc="mean",
            grad_mult=1,
        )

    def forward(self, input, aux):
        # input: [B, N(input_size), L]
        # aux: [B, N(input_size), L]

        B, _, L = input.size()

        # auxiliary
        aux = self._auxiliary(aux)
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

    def _auxiliary(self, aux):
        output = self.spk_encoder(aux)
        return output

class DPRNNRawNetTasNet(nn.Module):
    ''' DPRNN-RawNet-TasNet

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
        O: int.
        P: int.
        embeddings_size: int.
        num_spks: int.
        fusion_type: string, 'cat', 'add', 'mul' or 'film'.
    '''
    def __init__(self, input_size, feature_size=128, hidden_size=128,
                 chunk_length=200, kernel_size=2, hop_length=None,
                 n_repeats=6, bidirectional=True, rnn_type='LSTM',
                 norm_type='gLN', activation_type='sigmoid', dropout=0,
                 stride=None, O=128, P=256, embeddings_size=128,
                 num_spks=251, fusion_type='cat'):
        super().__init__()
        self.stride = stride if stride is not None else kernel_size // 2
        self.encoder = Encoder(
            kernel_size,
            input_size,
            stride=self.stride,
            bias=False,
        )
        self.separation = DPRNNRawNet(
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
            O,
            P,
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

    def forward(self, input, aux):
        # input: [B, L]
        # aux: [B, L]
        encoders = self.encoder(input)
        # -> [B, N, L]
        masks, aux = self.separation(encoders, aux)
        # -> [B, 2, N, L], [B, num_spks]
        output = masks * encoders.unsqueeze(1)
        # -> [B, 2, N, L]
        mixture = self.decoder(output[:, 0, :, :])
        # [B, L] (first speaker only)
        return mixture, aux
