import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=512):
        super().__init__()

        pe = torch.zeros(max_length, d_model)

        # creates position vector where every position has a distinct pattern(nearby pos's have similar patterns)
        for pos in range(max_length):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i/d_model))
                pe[pos, i] = math.sin(angle)

                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(angle)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length, :]

                


class TextTransformerEncoder(nn.Module):
    def __init__(
        self, d_model=128, num_heads = 4, num_layers = 2, d_ff= 256, dropout = 0.1
        ):

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
