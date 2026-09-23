"""Small White-positive centipawn API for engine-side adapters."""

from pathlib import Path

import chess
import torch

from .encoding import boards_to_batch
from .model import RayGNN, RayGNNConfig


class RayGNNEvaluator:
    def __init__(self, model: RayGNN, device: str | torch.device = "cpu"):
        self.device = device
        self.model = model.to(device).eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "RayGNNEvaluator":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = RayGNN(RayGNNConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        return cls(model, device)

    @torch.inference_mode()
    def evaluate_cp(self, boards: list[chess.Board]) -> torch.Tensor:
        """Return [B] White-positive centipawns; caller handles terminal positions."""
        batch = boards_to_batch(boards, self.device)
        return (self.model(batch).value[:, 0] * 100).cpu()
