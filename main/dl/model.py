from __future__ import annotations

import math

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        *,
        use_attention: bool = False,
        use_deep_head: bool = False,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_attention = bool(use_attention)
        self.use_deep_head = bool(use_deep_head)
        self.head_dropout = float(head_dropout)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        if self.use_attention:
            self.pre_head_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.pre_head_norm = nn.LayerNorm(hidden_size)
        else:
            self.pre_head_proj = None
            self.pre_head_norm = None

        if self.use_deep_head:
            h2 = max(hidden_size // 2, 1)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, h2),
                nn.LayerNorm(h2),
                nn.ReLU(),
                nn.Dropout(self.head_dropout),
                nn.Linear(h2, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, (h_n, _c_n) = self.lstm(x_seq)
        # h_n: (num_layers, batch, hidden); last layer state = query for attention
        last_h = h_n[-1]

        if self.use_attention:
            scale = math.sqrt(self.hidden_size)
            scores = torch.bmm(out, last_h.unsqueeze(-1)).squeeze(-1) / scale
            weights = torch.softmax(scores, dim=1)
            context = (weights.unsqueeze(-1) * out).sum(dim=1)
            combined = torch.cat([context, last_h], dim=-1)
            feat = self.pre_head_norm(self.pre_head_proj(combined))
        else:
            feat = out[:, -1, :]

        return self.head(feat).squeeze(-1)
