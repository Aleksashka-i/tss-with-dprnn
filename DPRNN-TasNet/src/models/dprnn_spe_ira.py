import torch
from torch import nn
from src.models.encoder_decoder import Encoder, Decoder
from src.models import norms
from src.models.dprnn_spe import ResBlock, DPRNNSpe

class DPRNNSpeIRA(DPRNNSpe):
    def __init__(self, input_size, output_size=None, feature_size=128,
                 hidden_size=128, chunk_length=200, hop_length=None,
                 n_repeats=6, bidirectional=True, rnn_type='LSTM',
                 norm_type='gLN', activation_type='sigmoid', dropout=0,
                 mid_resnet_size=256, embeddings_size=128, num_spks=251,
                 kernel_size=2):
        assert input_size == output_size, "Input size must be equal to output size (sorry)."
        super().__init__(
            input_size,
            output_size,
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
        )
        self.spk_encoder = nn.Sequential(
            norms.GlobLN(input_size),
            nn.Conv1d(input_size, feature_size, 1),
            ResBlock(feature_size, feature_size),
            ResBlock(feature_size, mid_resnet_size),
            ResBlock(mid_resnet_size, mid_resnet_size),
            nn.Conv1d(mid_resnet_size, embeddings_size, 1),
        )
        self.embeddings_size = embeddings_size
        self.aux_linear = nn.Linear(2 * embeddings_size, embeddings_size)

    def forward(self, input, aux, aux_len):
        ''' input: [B, N(input_size), L]
            aux: [B, N(input_size), L] 
        '''
        # first auxilary run
        v0 = self.auxilary(aux, aux_len)
        B, _, L = input.size()
        output_norm = self.bottleneck[0](input)
        # -> [B, N(input_size), L]
        # first extraction concatenation
        output = self.concatenation(v0, output_norm, L)
        output = self.bottleneck[1](output)
        # first estimated representation
        masks = self.dprnn_process(output, B, L)
        d0 = masks * input.unsqueeze(1)
        d0 = d0[:, 0, :, :]

        # second auxilary run
        v1 = self.auxilary(d0, aux_len)
        # auxilary concatenation
        v1 = torch.cat((v0, v1), dim=1) 
        # -> [B, 2 * embeddings_size]
        v1 = self.aux_linear(v1)
        # second extraction concatenation
        output = self.concatenation(v1, output_norm, L)
        output = self.bottleneck[1](output)
        # second estimated representation
        masks = self.dprnn_process(output, B, L)
        d1 = masks * input.unsqueeze(1)
        d1 = d1[:, 0, :, :]

        v1 = self.pred_linear(v1)
        return d1, v1
    
    def auxilary(self, aux, aux_len):
        ''' aux: [B, N(input_size), L] 
        '''
        aux = self.spk_encoder(aux)
        # -> [B, embeddings_size, L // 3]
        aux_T = (aux_len - self.kernel_size) // (self.kernel_size // 2) + 1
        aux_T = ((aux_T // 3) // 3) // 3
        aux = torch.sum(aux, -1) / aux_T.view(-1, 1).float()
        # -> [B, embeddings_size]
        return aux

    def concatenation(self, aux, output, L):
        # concatenation
        aux_concat = torch.unsqueeze(aux, -1)
        aux_concat  = aux_concat.repeat(1, 1, L)
         # -> [B, N(embeddings_size), L]
        output = torch.cat([output, aux_concat], 1)
        # -> [B, N(input_size + embeddings_size), L]
        return output

    def dprnn_process(self, output, B, L):
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
        # -> [2 * B, N(output_size), L]
        output = self.activation(output)
        output = output.reshape(B, 2, self.output_size, L)
        # -> [B, 2, N(output_size), L]
        return output # -> [B, N(output_size), L]

class DPRNNSpeIRATasNet(nn.Module):
    ''' DPRNN-TasNet

        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int.
        feature_size: int, default: 128.
        hidden_size: int, default: 128.
        kernel_size: int, default: 2(?).
        chunk_length: int, xw
        hop_length: int,
        n_repeats: int, default: 6.
        dropout: float, default: 0.
        bidirectional: bool, default: False.
        stride: int.
        norm_type: str.
        activation_type: str.
        mid_resnet_size: int.
        embeddings_size: int.
        num_spks: int.
    '''
    def __init__(self, input_size, output_size=None, feature_size=128,
                 hidden_size=128, chunk_length=200, kernel_size=2,
                 hop_length=None, n_repeats=6, bidirectional=True,
                 rnn_type='LSTM', norm_type='gLN', activation_type='sigmoid',
                 dropout=0, stride=None, mid_resnet_size=256,
                 embeddings_size=128, num_spks=251):
        super().__init__()
        self.output_size = output_size if output_size is not None else input_size
        self.stride = stride if stride is not None else kernel_size // 2
        self.encoder = Encoder(
            kernel_size,
            input_size,
            stride=self.stride,
            bias=False,
        )
        self.separation = DPRNNSpeIRA(
            input_size,
            output_size,
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
        )
        self.decoder = Decoder(
            in_channels=output_size,
            out_channels=1,
            kernel_size=kernel_size,
            stride=self.stride,
            bias=False,
        )

    def forward(self, input, aux, aux_len):
        ''' input: [B, L],
            aux: [B, L],
        '''
        encoders = self.encoder(input) # -> [B, N, L]
        embeddings = self.encoder(aux) # -> [B, N, L]
        output, aux = self.separation(encoders, embeddings, aux_len) # -> [B, N, L], [B, num_spks]
        mixture = self.decoder(output) # [B, L]
        return mixture, aux
