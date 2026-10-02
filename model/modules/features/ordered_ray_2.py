import torch
from torch import nn

from .input_feature import InputFeature


class OrderedRay2(InputFeature):
    """Two-occupied-square slider-ray relationships.

    The 41,472 collision-free keys encode slider colour/type/direction, the
    first and second pieces on the ray, and bucketed distances between them.
    """

    HASH = 0xA8D12F37
    FEATURE_NAME = "OrderedRay2"
    INPUT_FEATURE_NAME = "OrderedRay2"
    MAX_ACTIVE_FEATURES = 240

    NUM_INPUTS = 2 * 16 * 12 * 12 * 3 * 3
    NUM_REAL_FEATURES = NUM_INPUTS
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )
        # ReZero-style scalar. The ray table can have useful nonzero gradients
        # while its initial contribution to the parent network is exactly zero.
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight * self.gate

    @torch.no_grad()
    def coalesce(self) -> None:
        pass

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass

    @torch.no_grad()
    def init_weights(self) -> None:
        pass

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        # Stockfish needs no gate-aware file format: fold it into the table.
        return (self.weight.data * self.gate.data).clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)
        self.gate.data.fill_(1.0)

    def clip_weights(self, quantization) -> None:
        self.weight.data.clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )
