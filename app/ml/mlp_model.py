"""Shared PyTorch MLP architecture for 63-feature ASL hand landmark classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class ASLLandmarkMLP(nn.Module):
    """Compact MLP classifier over 63 normalized MediaPipe hand landmark coordinates."""

    def __init__(
        self,
        input_dim: int = 63,
        num_classes: int = 24,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
