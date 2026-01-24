import torch
import torch.nn as nn


class ShortcutConvBlock1D(nn.Module):
    """
    paper used: https://pmc.ncbi.nlm.nih.gov/articles/PMC7348856/?utm_source=chatgpt.com
    Paper-style shortcut conv block:
      BN -> Conv1D -> ReLU -> Dropout -> (AvgPool main + MaxPool shortcut on part of channels)

    "Max pool layer was used as a shortcut connection, which processes a part of the transmitted data"
    Block components in paper: BN, 1D-CNN, ReLU, Dropout, 1D AvgPool, 1D MaxPool.
    """
    def __init__(self, in_ch, out_ch, kernel_size=10, dropout=0.3, pool_kernel=2):
        super().__init__()
        if out_ch % 2 != 0:
            raise ValueError("out_ch must be even to split channels into main/shortcut parts.")

        self.bn = nn.BatchNorm1d(in_ch)
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # keep length before pooling
            bias=False,
        )
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        # Main vs shortcut pooling (paper-specific)
        self.avgpool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        self.maxpool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)

    def forward(self, x):
        # x: (B, C_in, T)
        x = self.bn(x)
        x = self.conv(x)     # (B, C_out, T)
        x = self.act(x)
        x = self.drop(x)

        # Shortcut processes "part" of transmitted data => split channels
        c_half = x.size(1) // 2
        x_main = x[:, :c_half, :]     # first half channels (main)
        x_short = x[:, c_half:, :]    # second half channels (shortcut)

        # Main uses AvgPool, shortcut uses MaxPool
        x_main = self.avgpool(x_main)     # (B, C_out/2, T/2)
        x_short = self.maxpool(x_short)   # (B, C_out/2, T/2)

        # Concatenate back (paper-style fusion)
        return torch.cat([x_main, x_short], dim=1)  # (B, C_out, T/2)


class CNN_LSTM_ECG(nn.Module):
    """
    8CSL adapted to my pipeline:
      - Accepts (in_channels, num_classes)
      - Input:  (B, C, T)
      - Output: (B, num_classes)

    Paper-style:
      - 8 CNN layers (8 shortcut-conv blocks)
      - 1-layer LSTM after CNN
      - Simple temporal pooling (mean) before FC
    """
    def __init__(self, in_channels, num_classes, dropout=0.3):
        super().__init__()

        # 8 CNN layers (keep channels EVEN for clean splitting)
        # You can tweak these, but the "8 blocks + shortcut pooling + k=10" is the paper-critical part.
        ch = [32, 32, 64, 64, 128, 128, 256, 256]

        blocks = []
        c_in = in_channels
        for c_out in ch:
            blocks.append(
                ShortcutConvBlock1D(
                    in_ch=c_in,
                    out_ch=c_out,
                    kernel_size=10,   # paper uses 10x1 filter
                    dropout=dropout,
                    pool_kernel=2
                )
            )
            c_in = c_out

        self.cnn = nn.Sequential(*blocks)

        # 1-layer LSTM (paper)
        self.lstm = nn.LSTM(
            input_size=ch[-1],
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0,   # no effect for 1 layer
            bidirectional=False
        )

        # FC classifier (paper ends with FC for classification)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.cnn(x)          # (B, 256, T')  (each block halves time => T' = T / 2^8)
        x = x.permute(0, 2, 1)   # (B, T', 256)

        out, _ = self.lstm(x)    # (B, T', 128)

        # Paper doesn't use attention; a stable choice is mean pooling
        out = out.mean(dim=1)    # (B, 128)

        return self.classifier(out)  # (B, num_classes)
