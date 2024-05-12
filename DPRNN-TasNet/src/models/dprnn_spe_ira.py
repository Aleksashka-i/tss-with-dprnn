import torch
from torch import nn
from src.models.encoder_decoder import Encoder, Decoder
from src.models.dprnn_spe import DPRNNSpe

class DPRNNSpeIRA(DPRNNSpe):
    ''' Dual-Path RNN. (Spe-IRA)

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
        P: int.
        embeddings_size: int.
        num_spks: int.
        kernel_size: int.
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
        self.aux_linear = nn.Linear(2 * embeddings_size, embeddings_size)

    def forward(self, input, aux, aux_len):
        # input: [B, N(input_size), L]
        # aux: [B, N(input_size), L]

        B, _, L = input.size()

        # 1st auxiliary
        v0 = self._auxiliary(aux, aux_len)
        # -> [B, N(embeddings_size)]

        # norm
        output_norm = self.bottleneck[0](input)
        # -> [B, N(input_size), L]

        # 1st fusion
        output = self._fusion(v0, output_norm, L)

        # 1st 1x1 cnn
        output = self.bottleneck[1](output)
        # -> [B, N(feature_size), L]

        # 1st dprnn blocks
        masks = self._dprnn_process(output, B, L)
        # -> [B, 2, N(input_size), L]

        # get d0 (est representation of target)
        d0 = masks * input.unsqueeze(1)
        d0 = d0[:, 0, :, :]
        # -> [B, N(input_size), L]

        # 2nd auxiliary
        v1 = self._auxiliary(d0, aux_len)
        # -> [B, N(embeddings_size)]

        # auxiliary cat
        v1 = torch.cat((v0, v1), dim=1)
        # -> [B, 2 * N(embeddings_size)]

        # 1st auxiliary linear
        v1 = self.aux_linear(v1)
        # -> [B, N(embeddings_size)]

        # 2nd fusion
        output = self._fusion(v1, output_norm, L)

        # 2nd 1x1 cnn
        output = self.bottleneck[1](output)
        # -> [B, N(feature_size), L]

        # 2nd dprnn blocks
        masks = self._dprnn_process(output, B, L)
        # -> [B, 2, N(input_size), L]

        # get d1 (better representation of target)
        d1 = masks * input.unsqueeze(1)
        d1 = d1[:, 0, :, :]
        # -> [B, N(input_size), L]

        # 2nd auxiliary linear
        v1 = self.pred_linear(v1)
        # -> [B, num_spks]

        return d1, v1

class DPRNNSpeIRATasNet(nn.Module):
    ''' DPRNN-Spe-IRA-TasNet

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
        P: int.
        embeddings_size: int.
        num_spks: int.
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
        self.separation = DPRNNSpeIRA(
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

    def forward(self, input, aux, aux_len):
        # input: [B, L]
        # aux: [B, L]
        encoders = self.encoder(input)
         # -> [B, N, L]
        embeddings = self.encoder(aux)
        # -> [B, N, L]
        output, aux = self.separation(encoders, embeddings, aux_len)
        # -> [B, N, L], [B, num_spks]
        mixture = self.decoder(output)
        # [B, L]
        return mixture, aux
