import torch
from torch import nn

from .config import ModelConfig
from .modules import (
    ComposedFeatureTransformer,
    DualMovementAccumulator,
    LayerStacks,
    MovementEvaluationNetwork,
    MovementFeatureDecoder,
    RayGNNEvaluationNetwork,
    RayGNNPosition,
    get_feature_cls,
)
from .modules.features.halfka_v2_hm import HalfKav2Hm
from .quantize import QuantizationManager


class NNUEModel(nn.Module):
    # Keeps older pickled .pt models, which predate this instance attribute,
    # on the conventional evaluator path.
    network_type = "nnue"

    def __init__(
        self,
        feature_name: str,
        config: ModelConfig,
        num_psqt_buckets: int = 8,
        num_ls_buckets: int = 8,
    ):
        super().__init__()

        self.network_type = config.network_type
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3

        self.quantize_config = config.quantize_config
        self.quantization = QuantizationManager(config.quantize_config)

        self.num_psqt_buckets = num_psqt_buckets
        self.num_ls_buckets = num_ls_buckets

        if self.network_type == "raygnn":
            self.input = None
            self.movement = None
            self.raygnn = RayGNNEvaluationNetwork(
                layers=config.raygnn_layers, with_wdl=config.raygnn_wdl
            )
            self.layer_stacks = None
            self.weight_clipping = []
            self.feature_name = "RayGNN-FEN"
            self.input_feature_name = "RayGNN-FEN"
            self.feature_hash = 0
        elif self.network_type == "movement":
            configured_features = get_feature_cls(feature_name)
            if sum(fc is HalfKav2Hm for fc in configured_features) != 1:
                raise ValueError(
                    "Movement networks require a HalfKAv2_hm^ feature component."
                )
            # Other configured feature components are redundant: movement
            # relationships replace threat/pawn-pair tables.  Asking the native
            # loader for HalfKA alone avoids extracting and transferring them.
            self.input = MovementFeatureDecoder(HalfKav2Hm.FEATURE_NAME)
            self.movement = MovementEvaluationNetwork(
                dim=config.movement_dim,
                iterations=config.movement_iterations,
            )
            self.layer_stacks = None
            self.raygnn = None
            self.weight_clipping = []
        else:
            feature_cls = get_feature_cls(feature_name)
            self.input = ComposedFeatureTransformer(
                feature_cls,
                self.L1,
                self.num_psqt_buckets,
                self.quantization,
            )
            self.movement = None
            self.raygnn = None
            self.layer_stacks = LayerStacks(
                self.num_ls_buckets, config, self.quantization
            )
            self.weight_clipping = self.quantization.generate_weight_clipping_config(
                self
            )
            self.input.init_weights()

        if self.input is not None:
            self.feature_name = self.input.FEATURE_NAME
            self.input_feature_name = self.input.INPUT_FEATURE_NAME
            self.feature_hash = self.input.HASH

    @torch.no_grad()
    def clip_weights(self, include_input):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        if include_input and self.input is not None:
            self.input.clip_weights(self.quantization)

        for group in self.weight_clipping:
            for p in group["params"]:
                if "min_weight" in group or "max_weight" in group:
                    p_data_fp32 = p.data
                    min_weight = group["min_weight"]
                    max_weight = group["max_weight"]
                    if "virtual_params" in group:
                        virtual_params = group["virtual_params"]
                        xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                        ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                        expanded_virtual_layer = virtual_params.repeat(xs, ys)
                        if min_weight is not None:
                            min_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                                - expanded_virtual_layer
                            )
                        if max_weight is not None:
                            max_weight = (
                                p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                                - expanded_virtual_layer
                            )
                    p_data_fp32.clamp_(min_weight, max_weight)


    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        if self.input is not None:
            self.input.zero_virtual_weights()
        if self.layer_stacks is not None:
            self.layer_stacks.zero_virtual_weights()


    def forward_ft(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        psqt_indices: torch.Tensor,
        fake_quantize_acts: bool,
        fake_quantize_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.network_type == "movement":
            raise RuntimeError("Movement networks do not use a feature transformer.")
        return self.input(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )

    def calculate_buckets(self, piece_count: torch.Tensor):
        psqt_indices = (piece_count - 1) // 4
        layer_stack_indices = psqt_indices

        return psqt_indices, layer_stack_indices


    def forward(
        self,
        us: torch.Tensor,
        them: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
        piece_count: torch.Tensor,
        fake_quantize_acts: bool=True,
        fake_quantize_weights: bool=True,
    ):
        if self.network_type == "raygnn":
            raise RuntimeError(
                "RayGNN requires fixed-White FEN/state input. Use forward_fens() or "
                "forward_position(); the sparse NNUE loader does not provide castling, "
                "en-passant, halfmove, or repetition state."
            )
        if self.network_type == "movement":
            _ = them, piece_count, fake_quantize_acts, fake_quantize_weights
            board = self.input.decode(us, white_indices, black_indices)
            return self.movement(board)

        psqt_indices, layer_stack_indices = self.calculate_buckets(piece_count)

        l0_, wpsqt, bpsqt = self.forward_ft(
            us,
            them,
            white_indices,
            black_indices,
            psqt_indices,
            fake_quantize_acts,
            fake_quantize_weights,
        )
        # The PSQT values are averaged over perspectives. "Their" perspective
        # has a negative influence (us-0.5 is 0.5 for white and -0.5 for black,
        # which does both the averaging and sign flip for black to move)
        x = self.layer_stacks(l0_, layer_stack_indices, fake_quantize_acts, fake_quantize_weights) + (wpsqt - bpsqt) * (us - 0.5)

        return x

    def forward_position(self, position: RayGNNPosition, return_wdl: bool = False):
        if self.raygnn is None:
            raise RuntimeError("forward_position is available only for RayGNN models.")
        return self.raygnn(position, return_wdl=return_wdl)

    def forward_fens(self, fens, repetition_count=0, return_wdl: bool = False):
        if self.raygnn is None:
            raise RuntimeError("forward_fens is available only for RayGNN models.")
        position = RayGNNPosition.from_fens(
            fens, repetition_count=repetition_count,
            device=next(self.raygnn.parameters()).device,
        )
        return self.raygnn(position, return_wdl=return_wdl)

    def forward_board(self, board: torch.Tensor) -> torch.Tensor:
        """Evaluate side-to-move-normalized piece codes directly."""
        if self.movement is None:
            raise RuntimeError("forward_board is available only for movement networks.")
        return self.movement(board)

    @torch.no_grad()
    def create_movement_accumulator(
        self,
        us: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
    ) -> DualMovementAccumulator:
        if self.movement is None:
            raise RuntimeError("Incremental state is available only for movement networks.")
        white_board = self.input.decode(
            torch.ones_like(us), white_indices, black_indices
        )
        black_board = self.input.decode(
            torch.zeros_like(us), white_indices, black_indices
        )
        white = self.movement.create_accumulator(white_board)
        black = self.movement.create_accumulator(black_board)
        white_to_move = bool((us[0, 0] >= 0.5).item())
        return DualMovementAccumulator(
            white=white,
            black=black,
            white_to_move=white_to_move,
            evaluation=white.evaluation if white_to_move else black.evaluation,
        )

    @torch.no_grad()
    def update_movement_accumulator(
        self,
        accumulator: DualMovementAccumulator,
        us: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
    ) -> DualMovementAccumulator:
        if self.movement is None:
            raise RuntimeError("Incremental state is available only for movement networks.")
        white_board = self.input.decode(
            torch.ones_like(us), white_indices, black_indices
        )
        black_board = self.input.decode(
            torch.zeros_like(us), white_indices, black_indices
        )
        white = self.movement.update_accumulator(accumulator.white, white_board)
        black = self.movement.update_accumulator(accumulator.black, black_board)
        white_to_move = bool((us[0, 0] >= 0.5).item())
        return DualMovementAccumulator(
            white=white,
            black=black,
            white_to_move=white_to_move,
            evaluation=white.evaluation if white_to_move else black.evaluation,
        )
