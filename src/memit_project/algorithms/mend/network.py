from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn


@dataclass
class MENDCheckpointMetadata:
    scale_candidates: List[float]
    feature_dim: int
    hidden_dim: int


class MENDGradientScaleNetwork(nn.Module):
    """
    Lightweight learned editor network for Qwen-style causal LMs.
    It maps per-layer gradient statistics to a categorical distribution over
    candidate update scales.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, n_candidates: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_candidates = n_candidates
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_candidates),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    @torch.no_grad()
    def predict_scales(
        self, features: torch.Tensor, scale_candidates: Iterable[float]
    ) -> torch.Tensor:
        candidates = torch.tensor(
            list(scale_candidates), device=features.device, dtype=features.dtype
        )
        logits = self(features)
        probs = torch.softmax(logits, dim=-1)
        return probs @ candidates
