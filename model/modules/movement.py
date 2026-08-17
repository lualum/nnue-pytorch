"""Tiny movement-geometry message passing evaluator.

The module deliberately uses edge lists and directional ray scans instead of a
64 x 64 attention tensor.  Piece locations are decoded from the HalfKAv2_hm
component already produced by the training data loader.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .features import get_feature_cls
from .features.halfka_v2_hm import HalfKav2Hm


EMPTY = 0
OUR_PAWN, THEIR_PAWN = 1, 2
OUR_KNIGHT, THEIR_KNIGHT = 3, 4
OUR_BISHOP, THEIR_BISHOP = 5, 6
OUR_ROOK, THEIR_ROOK = 7, 8
OUR_QUEEN, THEIR_QUEEN = 9, 10
OUR_KING, THEIR_KING = 11, 12

_ORTHOGONAL_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL_DIRECTIONS = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_RAY_DIRECTIONS = _ORTHOGONAL_DIRECTIONS + _DIAGONAL_DIRECTIONS
_KNIGHT_OFFSETS = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
_KING_OFFSETS = tuple(
    (file_delta, rank_delta)
    for file_delta in (-1, 0, 1)
    for rank_delta in (-1, 0, 1)
    if file_delta or rank_delta
)


def _square(file: int, rank: int) -> int:
    return rank * 8 + file


def _inside(file: int, rank: int) -> bool:
    return 0 <= file < 8 and 0 <= rank < 8


def _make_edges(
    offsets: tuple[tuple[int, int], ...],
    source_ranks: tuple[int, ...] = tuple(range(8)),
) -> tuple[torch.Tensor, torch.Tensor]:
    sources: list[int] = []
    destinations: list[int] = []
    for rank in source_ranks:
        for file in range(8):
            source = _square(file, rank)
            for file_delta, rank_delta in offsets:
                to_file = file + file_delta
                to_rank = rank + rank_delta
                if _inside(to_file, to_rank):
                    sources.append(source)
                    destinations.append(_square(to_file, to_rank))
    return torch.tensor(sources), torch.tensor(destinations)


def _make_double_pawn_edges(
    source_rank: int, rank_delta: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sources = torch.tensor([_square(file, source_rank) for file in range(8)])
    middles = torch.tensor(
        [_square(file, source_rank + rank_delta) for file in range(8)]
    )
    destinations = torch.tensor(
        [_square(file, source_rank + 2 * rank_delta) for file in range(8)]
    )
    return sources, middles, destinations


def _make_ray_lines(file_delta: int, rank_delta: int) -> torch.Tensor:
    """Return directional board lines as [depth, line], padded with -1."""
    starts: list[tuple[int, int]] = []
    for rank in range(8):
        for file in range(8):
            if not _inside(file - file_delta, rank - rank_delta):
                starts.append((file, rank))

    lines: list[list[int]] = []
    for start_file, start_rank in starts:
        line: list[int] = []
        file, rank = start_file, start_rank
        while _inside(file, rank):
            line.append(_square(file, rank))
            file += file_delta
            rank += rank_delta
        lines.append(line)

    result = torch.full((8, len(lines)), -1, dtype=torch.long)
    for line_index, line in enumerate(lines):
        result[: len(line), line_index] = torch.tensor(line)
    return result


def _feature_hash(feature_classes: list[type]) -> int:
    value = 0
    for feature_class in feature_classes:
        value = ((value << 1) | (value >> 31)) & 0xFFFFFFFF
        value ^= feature_class.HASH
    return value


class MovementFeatureDecoder(nn.Module):
    """Decode a side-to-move-normalized 64-square board from sparse features."""

    def __init__(self, feature_name: str):
        super().__init__()
        feature_classes = get_feature_cls(feature_name)

        offsets: list[tuple[type, int]] = []
        offset = 0
        for feature_class in feature_classes:
            offsets.append((feature_class, offset))
            offset += feature_class.NUM_INPUTS

        halfka_offsets = [
            component_offset
            for feature_class, component_offset in offsets
            if feature_class is HalfKav2Hm
        ]
        if len(halfka_offsets) != 1:
            raise ValueError(
                "The movement network requires exactly one HalfKAv2_hm^ "
                "feature component."
            )

        self.halfka_offset = halfka_offsets[0]
        self.NUM_INPUTS = sum(fc.NUM_INPUTS for fc in feature_classes)
        self.MAX_ACTIVE_FEATURES = sum(
            fc.MAX_ACTIVE_FEATURES for fc in feature_classes
        )
        self.NUM_REAL_FEATURES = sum(fc.NUM_REAL_FEATURES for fc in feature_classes)
        self.FEATURE_NAME = "+".join(fc.FEATURE_NAME for fc in feature_classes)
        self.INPUT_FEATURE_NAME = "+".join(
            fc.INPUT_FEATURE_NAME for fc in feature_classes
        )
        self.HASH = _feature_hash(feature_classes)

    def decode(
        self,
        us: torch.Tensor,
        white_indices: torch.Tensor,
        black_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return piece codes [batch, 64], normalized to the moving side.

        HalfKAv2_hm already flips Black vertically and mirrors around the
        perspective king.  Selecting the moving side's feature row therefore
        makes "our" pawns always move toward increasing ranks and gives the
        readout a natural negamax orientation.
        """
        if white_indices.shape != black_indices.shape:
            raise ValueError("White and black sparse feature tensors must match.")
        if us.ndim != 2 or us.shape[1] != 1:
            raise ValueError(f"Expected us shape [batch, 1], got {tuple(us.shape)}.")

        selected = torch.where(us >= 0.5, white_indices, black_indices).long()
        local = selected - self.halfka_offset
        valid = (local >= 0) & (local < HalfKav2Hm.NUM_INPUTS)

        within_bucket = torch.remainder(local.clamp_min(0), HalfKav2Hm.NUM_PLANES)
        squares = torch.remainder(within_bucket, 64)
        piece_codes = torch.div(within_bucket, 64, rounding_mode="floor") + 1

        # Index 64 is a scratch destination for all padding and other composed
        # feature components.  Trimming it avoids invalid entries overwriting a1.
        scratch = torch.zeros(
            selected.shape[0], 65, dtype=torch.long, device=selected.device
        )
        destinations = torch.where(valid, squares, torch.full_like(squares, 64))
        values = torch.where(valid, piece_codes, torch.zeros_like(piece_codes))
        return scratch.scatter(1, destinations, values)[:, :64]

    # Compatibility no-ops for trainer callbacks.  Legacy .nnue export is
    # rejected explicitly by NNUEWriter because it cannot represent this graph.
    def clip_weights(self, quantization) -> None:
        _ = quantization

    def coalesce(self) -> None:
        pass

    def init_weights(self) -> None:
        pass

    def zero_virtual_weights(self) -> None:
        pass

    def get_export_weights(self) -> torch.Tensor:
        raise RuntimeError("Movement networks do not have feature-transformer weights.")


class MovementUpdate(nn.Module):
    """A shared, minimal gated update used at every reasoning iteration."""

    def __init__(self, dim: int):
        super().__init__()
        self.self_candidate = nn.Linear(dim, dim)
        self.message_candidate = nn.Linear(dim, dim, bias=False)
        self.state_candidate = nn.Linear(dim, dim, bias=False)
        self.self_gate = nn.Linear(dim, dim)
        self.message_gate = nn.Linear(dim, dim, bias=False)

    def forward(
        self, hidden: torch.Tensor, messages: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        candidate = torch.clamp(
            self.self_candidate(hidden)
            + self.message_candidate(messages)
            + self.state_candidate(state),
            -1.0,
            1.0,
        )
        gate = torch.clamp(
            0.2 * (self.self_gate(hidden) + self.message_gate(messages)) + 0.5,
            0.0,
            1.0,
        )
        return hidden + gate * (candidate - hidden)


@dataclass(frozen=True)
class MovementAccumulator:
    """Cached inference state for one position."""

    board: torch.Tensor
    states: tuple[torch.Tensor, ...]
    evaluation: torch.Tensor
    updated_squares: tuple[int, ...]


@dataclass(frozen=True)
class DualMovementAccumulator:
    """White/Black perspective caches, selected by side to move at readout."""

    white: MovementAccumulator
    black: MovementAccumulator
    white_to_move: bool
    evaluation: torch.Tensor


class MovementEvaluationNetwork(nn.Module):
    """Small shared-weight graph network over chess movement relationships."""

    def __init__(self, dim: int = 8, iterations: int = 3):
        super().__init__()
        if not 3 <= iterations <= 5:
            raise ValueError("iterations must be between 3 and 5.")

        self.dim = dim
        self.iterations = iterations

        self.piece_embedding = nn.Embedding(13, dim)
        self.square_embedding = nn.Embedding(64, dim)

        self.knight_message = nn.Linear(dim, dim)
        self.king_message = nn.Linear(dim, dim)
        self.pawn_diagonal_message = nn.Linear(dim, dim)
        self.pawn_forward_message = nn.Linear(dim, dim)
        self.ray_message = nn.Linear(dim, dim)

        # Ordered movement paths share this learned state-dependent transition.
        # It is linear in the carried message, preserving additive contributions.
        self.path_gate = nn.Linear(dim, dim)
        self.path_scale = nn.Parameter(torch.zeros(dim))
        self.update = MovementUpdate(dim)

        self.readout_hidden = nn.Linear(dim, dim)
        self.readout = nn.Linear(dim, 1)

        self._register_geometry()
        self._dirty_dependents = self._make_dirty_dependents()
        self.reset_parameters()

    def _register_geometry(self) -> None:
        edges = {
            "knight": _make_edges(_KNIGHT_OFFSETS),
            "king": _make_edges(_KING_OFFSETS),
            "our_pawn_diagonal": _make_edges(((-1, 1), (1, 1)), tuple(range(1, 7))),
            "their_pawn_diagonal": _make_edges(
                ((-1, -1), (1, -1)), tuple(range(1, 7))
            ),
            "our_pawn_step": _make_edges(((0, 1),), tuple(range(1, 7))),
            "their_pawn_step": _make_edges(((0, -1),), tuple(range(1, 7))),
        }
        for name, (sources, destinations) in edges.items():
            self.register_buffer(f"_{name}_sources", sources, persistent=False)
            self.register_buffer(
                f"_{name}_destinations", destinations, persistent=False
            )

        for name, source_rank, rank_delta in (
            ("our_pawn_double", 1, 1),
            ("their_pawn_double", 6, -1),
        ):
            sources, middles, destinations = _make_double_pawn_edges(
                source_rank, rank_delta
            )
            self.register_buffer(f"_{name}_sources", sources, persistent=False)
            self.register_buffer(f"_{name}_middles", middles, persistent=False)
            self.register_buffer(
                f"_{name}_destinations", destinations, persistent=False
            )

        for direction_index, (file_delta, rank_delta) in enumerate(_RAY_DIRECTIONS):
            self.register_buffer(
                f"_ray_lines_{direction_index}",
                _make_ray_lines(file_delta, rank_delta),
                persistent=False,
            )

    @staticmethod
    def _make_dirty_dependents() -> tuple[tuple[int, ...], ...]:
        """Conservative dependency lists for exact incremental cache updates."""
        dependents: list[set[int]] = [{square} for square in range(64)]
        jump_offsets = _KNIGHT_OFFSETS + _KING_OFFSETS + (
            (-1, 1),
            (0, 1),
            (1, 1),
            (-1, -1),
            (0, -1),
            (1, -1),
        )
        for source in range(64):
            file, rank = source % 8, source // 8
            for file_delta, rank_delta in jump_offsets:
                to_file, to_rank = file + file_delta, rank + rank_delta
                if _inside(to_file, to_rank):
                    dependents[source].add(_square(to_file, to_rank))
            # Every state on an ordered ray can transform a carried message, so
            # every square sharing one of its four lines is conservatively dirty.
            for file_delta, rank_delta in _RAY_DIRECTIONS:
                to_file, to_rank = file + file_delta, rank + rank_delta
                while _inside(to_file, to_rank):
                    dependents[source].add(_square(to_file, to_rank))
                    to_file += file_delta
                    to_rank += rank_delta
        return tuple(tuple(sorted(squares)) for squares in dependents)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.piece_embedding.weight, std=0.1)
        nn.init.normal_(self.square_embedding.weight, std=0.05)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Begin with cautious recurrent changes and mostly transmitting paths.
        nn.init.constant_(self.update.self_gate.bias, -1.0)
        nn.init.constant_(self.path_gate.bias, -2.0)
        nn.init.zeros_(self.readout.bias)

    @staticmethod
    def _piece_mask(board: torch.Tensor, codes: tuple[int, ...]) -> torch.Tensor:
        mask = torch.zeros_like(board, dtype=torch.bool)
        for code in codes:
            mask = mask | (board == code)
        return mask

    @staticmethod
    def _edge_values(
        projected: torch.Tensor,
        source_mask: torch.Tensor,
        sources: torch.Tensor,
        destinations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = projected.index_select(1, sources)
        mask = source_mask.index_select(1, sources)
        return values * mask.unsqueeze(-1), destinations

    def _path_factors(self, square_state: torch.Tensor) -> torch.Tensor:
        gate = torch.clamp(0.2 * self.path_gate(square_state) + 0.5, 0.0, 1.0)
        scale = torch.clamp(self.path_scale, -1.0, 1.0)
        return 1.0 + gate * (scale - 1.0)

    def _ray_values(
        self,
        path_factors: torch.Tensor,
        projected: torch.Tensor,
        slider_mask: torch.Tensor,
        lines: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch_size = path_factors.shape[0]
        line_count = lines.shape[1]
        carrier = path_factors.new_zeros(batch_size, line_count, self.dim)
        values: list[torch.Tensor] = []
        destinations: list[torch.Tensor] = []

        for depth in range(lines.shape[0]):
            square_indices = lines[depth]
            valid = square_indices >= 0
            safe_indices = square_indices.clamp_min(0)
            valid_values = valid.view(1, -1, 1)

            # Every square receives the current carrier. Its learned state then
            # transforms the carrier before any message originating here is added.
            values.append(carrier * valid_values)
            destinations.append(safe_indices)

            current_factors = path_factors.index_select(1, safe_indices)
            is_slider = slider_mask.index_select(1, safe_indices) & valid.view(1, -1)
            seed = projected.index_select(1, safe_indices) * is_slider.unsqueeze(-1)
            carrier = (carrier * current_factors + seed) * valid_values

        return values, destinations

    def aggregate_messages(
        self, hidden: torch.Tensor, board: torch.Tensor
    ) -> torch.Tensor:
        """Add incoming movement messages without constructing dense attention."""
        if hidden.shape[:2] != board.shape or hidden.shape[2] != self.dim:
            raise ValueError("Hidden state and board shapes do not match.")

        messages = hidden.new_zeros(hidden.shape)

        knight_mask = self._piece_mask(board, (OUR_KNIGHT, THEIR_KNIGHT))
        king_mask = self._piece_mask(board, (OUR_KING, THEIR_KING))
        our_pawn_mask = board == OUR_PAWN
        their_pawn_mask = board == THEIR_PAWN
        orthogonal_mask = self._piece_mask(
            board, (OUR_ROOK, THEIR_ROOK, OUR_QUEEN, THEIR_QUEEN)
        )
        diagonal_mask = self._piece_mask(
            board, (OUR_BISHOP, THEIR_BISHOP, OUR_QUEEN, THEIR_QUEEN)
        )

        knight_projected = self.knight_message(hidden)
        king_projected = self.king_message(hidden)
        pawn_diagonal_projected = self.pawn_diagonal_message(hidden)
        pawn_forward_projected = self.pawn_forward_message(hidden)
        path_factors = self._path_factors(hidden)
        edge_specs = (
            (knight_projected, knight_mask, "knight"),
            (king_projected, king_mask, "king"),
            (
                pawn_diagonal_projected,
                our_pawn_mask,
                "our_pawn_diagonal",
            ),
            (
                pawn_diagonal_projected,
                their_pawn_mask,
                "their_pawn_diagonal",
            ),
            (
                pawn_forward_projected,
                our_pawn_mask,
                "our_pawn_step",
            ),
            (
                pawn_forward_projected,
                their_pawn_mask,
                "their_pawn_step",
            ),
        )
        for projected, source_mask, name in edge_specs:
            edge_values, edge_destinations = self._edge_values(
                projected,
                source_mask,
                getattr(self, f"_{name}_sources"),
                getattr(self, f"_{name}_destinations"),
            )
            messages.index_add_(1, edge_destinations, edge_values)

        for name, source_mask in (
            ("our_pawn_double", our_pawn_mask),
            ("their_pawn_double", their_pawn_mask),
        ):
            sources = getattr(self, f"_{name}_sources")
            middles = getattr(self, f"_{name}_middles")
            edge_destinations = getattr(self, f"_{name}_destinations")
            edge_values, edge_destinations = self._edge_values(
                pawn_forward_projected,
                source_mask,
                sources,
                edge_destinations,
            )
            edge_values = edge_values * path_factors.index_select(1, middles)
            messages.index_add_(1, edge_destinations, edge_values)

        ray_projected = self.ray_message(hidden)
        for direction_index in range(len(_RAY_DIRECTIONS)):
            orthogonal = direction_index < len(_ORTHOGONAL_DIRECTIONS)
            ray_values, ray_destinations = self._ray_values(
                path_factors,
                ray_projected,
                orthogonal_mask if orthogonal else diagonal_mask,
                getattr(self, f"_ray_lines_{direction_index}"),
            )
            messages.index_add_(
                1, torch.cat(ray_destinations), torch.cat(ray_values, dim=1)
            )

        return messages

    def _initial_state(self, board: torch.Tensor) -> torch.Tensor:
        square_indices = torch.arange(64, device=board.device)
        return torch.clamp(
            self.piece_embedding(board)
            + self.square_embedding(square_indices).unsqueeze(0),
            -1.0,
            1.0,
        )

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        return self.readout(torch.clamp(self.readout_hidden(pooled), -1.0, 1.0))

    def _all_states(self, board: torch.Tensor) -> tuple[torch.Tensor, ...]:
        state = self._initial_state(board)
        static_state = state
        states = [state]
        for _ in range(self.iterations):
            messages = self.aggregate_messages(state, board)
            state = self.update(state, messages, static_state)
            states.append(state)
        return tuple(states)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        if board.ndim != 2 or board.shape[1] != 64:
            raise ValueError(f"Expected board shape [batch, 64], got {tuple(board.shape)}.")
        if board.dtype not in (torch.int32, torch.int64):
            raise ValueError("Board piece codes must be integer tensors.")
        state = self._all_states(board)[-1]
        return self._pool(state)

    @torch.no_grad()
    def create_accumulator(self, board: torch.Tensor) -> MovementAccumulator:
        """Create a cached iterative state for one search position."""
        if board.ndim == 1:
            board = board.unsqueeze(0)
        if board.shape != (1, 64):
            raise ValueError("Incremental accumulators support one [64] position.")
        states = self._all_states(board)
        evaluation = self._pool(states[-1])
        return MovementAccumulator(
            board=board.clone(),
            states=tuple(state.clone() for state in states),
            evaluation=evaluation.clone(),
            updated_squares=(64,) * (self.iterations + 1),
        )

    def _expand_dirty(self, dirty: torch.Tensor) -> torch.Tensor:
        expanded = dirty.clone()
        for square in torch.nonzero(dirty[0], as_tuple=False).flatten().tolist():
            expanded[0, list(self._dirty_dependents[square])] = True
        return expanded

    @torch.no_grad()
    def update_accumulator(
        self, accumulator: MovementAccumulator, board: torch.Tensor
    ) -> MovementAccumulator:
        """Update only the conservative dependency cone of changed squares.

        This is an exact reference implementation for a Stockfish-side cache.
        The tiny PyTorch graph still forms aggregate candidates in a batch, but
        cached hidden vectors outside the movement/ray dependency cone are not
        replaced.  A native implementation can update the same indexed lists in
        place and avoid those candidate calculations as well.
        """
        if board.ndim == 1:
            board = board.unsqueeze(0)
        if board.shape != (1, 64) or board.device != accumulator.board.device:
            raise ValueError("Updated board must be one [64] position on the cache device.")

        dirty = board != accumulator.board
        if not torch.any(dirty):
            return MovementAccumulator(
                board=accumulator.board,
                states=accumulator.states,
                evaluation=accumulator.evaluation,
                updated_squares=(0,) * (self.iterations + 1),
            )

        fresh_static = self._initial_state(board)
        state = torch.where(dirty.unsqueeze(-1), fresh_static, accumulator.states[0])
        static_state = state
        states = [state]
        updated_counts = [int(dirty.sum().item())]

        for iteration in range(self.iterations):
            dirty = self._expand_dirty(dirty)
            messages = self.aggregate_messages(state, board)
            candidate = self.update(state, messages, static_state)
            state = torch.where(
                dirty.unsqueeze(-1), candidate, accumulator.states[iteration + 1]
            )
            states.append(state)
            updated_counts.append(int(dirty.sum().item()))

        evaluation = self._pool(state)
        return MovementAccumulator(
            board=board.clone(),
            states=tuple(states),
            evaluation=evaluation,
            updated_squares=tuple(updated_counts),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = [
    "DualMovementAccumulator",
    "MovementAccumulator",
    "MovementEvaluationNetwork",
    "MovementFeatureDecoder",
    "MovementUpdate",
    "EMPTY",
    "OUR_PAWN",
    "THEIR_PAWN",
    "OUR_KNIGHT",
    "THEIR_KNIGHT",
    "OUR_BISHOP",
    "THEIR_BISHOP",
    "OUR_ROOK",
    "THEIR_ROOK",
    "OUR_QUEEN",
    "THEIR_QUEEN",
    "OUR_KING",
    "THEIR_KING",
]
