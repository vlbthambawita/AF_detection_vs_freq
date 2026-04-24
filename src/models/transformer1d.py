import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        # x: (B, T, D)
        t = x.size(1)
        return x + self.pe[:t].unsqueeze(0)


class Transformer1D(nn.Module):
    """
    Pure 1D time-series Transformer encoder for ECG classification.

    Input:  (B, C, T)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.2,
        patch_size=10,
        patch_stride=5,
        max_len=20000,
    ):
        super().__init__()

        # Tokenize the signal along time with a strided 1D projection.
        self.patch_embed = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=patch_size // 2,
            bias=False,
        )
        self.input_norm = nn.BatchNorm1d(d_model)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.patch_embed(x)   # (B, D, T_tokens)
        x = self.input_norm(x)
        x = x.transpose(1, 2)     # (B, T_tokens, D)

        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.final_norm(x)

        # Global average over tokens for sequence-level classification.
        x = x.mean(dim=1)         # (B, D)
        return self.classifier(x)