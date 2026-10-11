"""Paper Sections 3.2–3.3; all new convolutions retain PyTorch initialization."""

import torch
from torch import nn


class ConvBNSiLU(nn.Sequential):
    def __init__(self, c1, c2, kernel=1, stride=1, padding=0, groups=1, activate=True):
        super().__init__(
            nn.Conv2d(c1, c2, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03),
            nn.SiLU(inplace=True) if activate else nn.Identity(),
        )


class DPE(nn.Module):
    """Image-level three-branch mixture with a learned unconstrained residual scale."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        if channels % reduction:
            raise ValueError("DPE channels must be divisible by reduction")
        width = channels // reduction
        self.reduce = ConvBNSiLU(channels, width)
        self.horizontal = ConvBNSiLU(width, width, (1, 5), padding=(0, 2))
        self.vertical = ConvBNSiLU(width, width, (5, 1), padding=(2, 0))
        self.local = ConvBNSiLU(width, width, 3, padding=1)
        self.expand = ConvBNSiLU(width, channels, activate=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 8, 8), 3, 1, bias=True),
            nn.Softmax(dim=1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        u = self.reduce(x)
        weights = self.gate(x)
        mixed = (
            weights[:, 0:1] * self.horizontal(u)
            + weights[:, 1:2] * self.vertical(u)
            + weights[:, 2:3] * self.local(u)
        )
        return x + self.alpha * self.expand(mixed)


class OCDF(nn.Module):
    """Enhanced P2 detail -> neck P3; the semantic tensor determines orientation."""

    def __init__(self, detail_channels=64, semantic_channels=64):
        super().__init__()
        self.horizontal = nn.Sequential(
            ConvBNSiLU(detail_channels, detail_channels, (1, 5), 2, (0, 2), detail_channels),
            ConvBNSiLU(detail_channels, semantic_channels),
        )
        self.vertical = nn.Sequential(
            ConvBNSiLU(detail_channels, detail_channels, (5, 1), 2, (2, 0), detail_channels),
            ConvBNSiLU(detail_channels, semantic_channels),
        )
        hidden = max(semantic_channels // 8, 8)
        self.orientation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(semantic_channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True),
            nn.Softmax(dim=1),
        )
        self.gate = nn.Sequential(
            ConvBNSiLU(2 * semantic_channels, semantic_channels, activate=False),
            nn.Sigmoid(),
        )
        self.beta = nn.Parameter(torch.tensor(0.02))

    def forward(self, inputs):
        p2, semantic = inputs
        weights = self.orientation(semantic)
        detail = weights[:, 0:1] * self.horizontal(p2) + weights[:, 1:2] * self.vertical(p2)
        if detail.shape != semantic.shape:
            raise ValueError(f"OCDF shape mismatch: detail={detail.shape}, semantic={semantic.shape}")
        return semantic + self.beta * self.gate(torch.cat((semantic, detail), dim=1)) * detail


class SemanticIdentity(nn.Module):
    """Keep topology stable when OCDF is disabled in an ablation."""

    def forward(self, inputs):
        return inputs[1]
