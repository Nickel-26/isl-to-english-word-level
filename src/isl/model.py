"""Sequence classifiers for normalized keypoint clips.

Two config-selectable models:

* ``bigru``      - LayerNorm -> BiGRU -> mean-pool -> dropout -> linear
* ``transformer``- linear projection + learned positions -> encoder -> mean-pool

Both consume ``(batch, 32, 137)`` and output ``(batch, num_classes)``.
Keep them small: 958 training clips and a 6 GB laptop GPU.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .config import Config, ConfigError


class BiGRUClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.input_norm(x))
        return self.head(out.mean(dim=1))


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        seq_len: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        nhead: int = 4,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.position.shape[1]:
            raise ValueError(
                f"Expected {self.position.shape[1]} frames, got {x.shape[1]}"
            )
        h = self.project(self.input_norm(x)) + self.position
        return self.head(self.encoder(h).mean(dim=1))


def build_model(cfg: Config) -> nn.Module:
    kind = str(cfg.get("model.type", "bigru")).lower()
    common = dict(
        input_dim=int(cfg.feature_dim),
        num_classes=int(cfg.num_classes),
        hidden_size=int(cfg.get("model.hidden_size", 128)),
        num_layers=int(cfg.get("model.num_layers", 2)),
        dropout=float(cfg.get("model.dropout", 0.3)),
    )
    if kind == "bigru":
        return BiGRUClassifier(**common)
    if kind == "transformer":
        return TransformerClassifier(
            **common, seq_len=int(cfg.get("keypoints.resample_frames", 32))
        )
    raise ConfigError(f"Unknown model.type: {kind!r} (expected 'bigru' or 'transformer')")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_from_checkpoint(payload: dict) -> nn.Module:
    """Rebuild the exact architecture recorded in a training checkpoint."""
    kind = str(payload.get("model_type", "bigru")).lower()
    labels = payload["labels"]
    dim = int(payload["feature_dim"])
    seq_len = int(payload.get("seq_len", 32))
    saved = payload.get("model_config", {})
    common = dict(
        input_dim=dim,
        num_classes=len(labels),
        hidden_size=int(saved.get("hidden_size", 128)),
        num_layers=int(saved.get("num_layers", 2)),
        dropout=float(saved.get("dropout", 0.3)),
    )
    if kind == "bigru":
        return BiGRUClassifier(**common)
    if kind == "transformer":
        return TransformerClassifier(**common, seq_len=seq_len)
    raise ConfigError(f"Unknown model_type in checkpoint: {kind!r}")


def load_checkpoint_model(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, dict]:
    """Load a checkpoint and return ``(model_with_weights, payload)``.

    Keeping this in one place avoids accidentally evaluating a freshly
    initialized model (a bug that is easy to miss because shapes still match).
    """
    payload = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location=device,
        weights_only=False,
    )
    model = model_from_checkpoint(payload).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
