# ================================================================
# FILE: models.py
# Exact copy of TCN from Kaggle Cell 16
# ================================================================

import torch
import torch.nn as nn
 
 
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.pad, dilation=dilation)
 
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out
 
 
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.4):
        super().__init__()
        self.conv1    = CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2    = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.act      = nn.GELU()
        self.dropout  = nn.Dropout(dropout)
        self.residual = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())
 
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.act(self.bn2(self.conv2(out)))
        return self.act(out + self.residual(x))
 
 
class TCN(nn.Module):
    def __init__(self, num_numerical, cat_dims, embed_dim=8,
                 num_channels=None, kernel_size=3,
                 dropout=0.4, num_classes=8):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 128, 256, 256]
 
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(dim + 1, embed_dim) for dim in cat_dims])
        input_dim = embed_dim * len(cat_dims) + num_numerical
        self.input_proj = nn.Conv1d(input_dim, num_channels[0], 1)
 
        blocks = []
        for i in range(len(num_channels) - 1):
            blocks.append(TCNBlock(
                num_channels[i], num_channels[i + 1],
                kernel_size, dilation=2 ** i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
 
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
 
    def forward(self, num_seq, cat_seq):
        B, T, _ = num_seq.shape
        cat_emb = torch.cat(
            [self.cat_embeddings[i](cat_seq[:, :, i])
             for i in range(len(self.cat_embeddings))], dim=-1)
        x = torch.cat([num_seq, cat_emb], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.tcn(x)
        x = x[:, :, -1]
        return self.classifier(x)
