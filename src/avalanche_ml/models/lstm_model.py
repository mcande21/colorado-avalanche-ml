"""Hierarchical multi-rate LSTM for avalanche danger prediction."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class HierarchicalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 98,
        hidden_size: int = 64,
        num_layers: int = 2,
        lstm_dropout: float = 0.2,
        trunk_dropout: float = 0.3,
        n_problem_types: int = 3,
        n_danger_levels: int = 5,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        lstm_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": lstm_dropout,
            "batch_first": True,
        }
        self.branch1_lstm = nn.LSTM(**lstm_kwargs)
        self.branch2_lstm = nn.LSTM(**lstm_kwargs)
        self.branch3_lstm = nn.LSTM(**lstm_kwargs)

        concat_size = hidden_size * 3
        self.trunk = nn.Sequential(
            nn.Linear(concat_size, 128),
            nn.ReLU(),
            nn.Dropout(trunk_dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(trunk_dropout),
        )

        self.stage1_head = nn.Linear(64, n_problem_types)

        self.stage2_binary_head = nn.Linear(64 + n_problem_types, 1)
        self.stage2_multi_head = nn.Linear(64 + n_problem_types, n_danger_levels)

    def forward(
        self,
        branch1: torch.Tensor,
        branch2: torch.Tensor,
        branch3: torch.Tensor,
        branch3_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, (h1, _) = self.branch1_lstm(branch1)
        h1 = h1[-1]

        _, (h2, _) = self.branch2_lstm(branch2)
        h2 = h2[-1]

        h3 = self._encode_branch3(branch3, branch3_lengths)

        combined = torch.cat([h1, h2, h3], dim=1)
        trunk_out = self.trunk(combined)

        problem_logits = self.stage1_head(trunk_out)
        problem_types = torch.sigmoid(problem_logits)

        stage2_input = torch.cat([trunk_out, problem_types], dim=1)
        danger_binary = torch.sigmoid(self.stage2_binary_head(stage2_input))
        danger_distribution = torch.softmax(
            self.stage2_multi_head(stage2_input), dim=1,
        )

        return {
            "problem_types": problem_types,
            "danger_binary": danger_binary,
            "danger_distribution": danger_distribution,
        }

    def _encode_branch3(
        self, branch3: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths_clamped = lengths.clamp(min=1)
        packed = pack_padded_sequence(
            branch3, lengths_clamped.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (h3, _) = self.branch3_lstm(packed)
        return h3[-1]
