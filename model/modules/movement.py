"""Tiny movement-geometry message passing evaluator.

The module uses sparse edge lists rather than a 64 x 64 attention tensor.
Sliding-piece relations are generated directly for every source/target pair;
they do not depend on a sequential ray scan.
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


def _make_ray_pairs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all directed slider relations and their intervening squares.

    ``paths[e]`` contains only squares strictly between a relation's source and
    target, in board order.  Each relation can therefore expose x-rays in one
    message-passing layer: a rook and a queen separated by a pawn are connected
    directly, while the pawn is included as the first blocker context.
    """
    sources: list[int] = []
    destinations: list[int] = []
    directions: list[int] = []
    distances: list[int] = []
    paths: list[list[int]] = []
    for source_rank in range(8):
        for source_file in range(8):
            source = _square(source_file, source_rank)
            for direction, (file_delta, rank_delta) in enumerate(_RAY_DIRECTIONS):
                path: list[int] = []
                file, rank = source_file + file_delta, source_rank + rank_delta
                distance = 1
                while _inside(file, rank):
                    sources.append(source)
                    destinations.append(_square(file, rank))
                    directions.append(direction)
                    distances.append(distance)
                    paths.append(path.copy())
                    path.append(_square(file, rank))
                    file += file_delta
                    rank += rank_delta
                    distance += 1
    padded = torch.full((len(paths), 6), -1, dtype=torch.long)
    for index, path in enumerate(paths):
        if path:
            padded[index, : len(path)] = torch.tensor(path)
    return (
        torch.tensor(sources),
        torch.tensor(destinations),
        torch.tensor(directions),
        torch.tensor(distances),
        padded,
    )


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

    # Kept only so already-created configs and checkpoints can be loaded. Rays
    # are direct relations in every current model.
    ordered_rays = False

    def __init__(self, dim: int = 8, iterations: int = 3, ordered_rays: bool = False):
        super().__init__()
        if not 3 <= iterations <= 5:
            raise ValueError("iterations must be between 3 and 5.")

        self.dim = dim
        self.iterations = iterations
        self.ordered_rays = ordered_rays

        self.piece_embedding = nn.Embedding(13, dim)
        self.square_embedding = nn.Embedding(64, dim)

        self.knight_message = nn.Linear(dim, dim)
        self.king_message = nn.Linear(dim, dim)
        self.pawn_diagonal_message = nn.Linear(dim, dim)
        self.pawn_forward_message = nn.Linear(dim, dim)
        self.ray_message = nn.Linear(dim, dim)
        self.ray_target = nn.Linear(dim, dim, bias=False)
        self.ray_first_blocker = nn.Linear(dim, dim, bias=False)
        self.ray_second_blocker = nn.Linear(dim, dim, bias=False)
        self.ray_direction = nn.Embedding(len(_RAY_DIRECTIONS), dim)
        self.ray_distance = nn.Embedding(8, dim)
        self.ray_blocker_count = nn.Embedding(4, dim)
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

        sources, destinations, directions, distances, paths = _make_ray_pairs()
        self.register_buffer("_ray_sources", sources, persistent=False)
        self.register_buffer("_ray_destinations", destinations, persistent=False)
        self.register_buffer("_ray_directions", directions, persistent=False)
        self.register_buffer("_ray_distances", distances, persistent=False)
        self.register_buffer("_ray_paths", paths, persistent=False)

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
            # A changed square can be source, target or blocker for every ray
            # relation sharing one of its ranks, files or diagonals.
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
        # Begin with cautious recurrent changes.
        nn.init.constant_(self.update.self_gate.bias, -1.0)
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

    def _ray_values(
        self,
        hidden: torch.Tensor,
        board: torch.Tensor,
        orthogonal_mask: torch.Tensor,
        diagonal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate every slider-to-square ray relation in parallel.

        A relation receives the source and target state, the first two occupied
        squares strictly between them, direction, distance, and a clipped
        blocker count. ``min`` reductions select blockers from fixed path
        tensors, rather than carrying a message through one square at a time.
        """
        paths = self._ray_paths
        safe_paths = paths.clamp_min(0)
        path_valid = paths.ge(0).view(1, -1, paths.shape[1])
        path_codes = board.index_select(1, safe_paths.flatten()).view(
            board.shape[0], paths.shape[0], paths.shape[1]
        )
        occupied = path_valid & path_codes.ne(EMPTY)
        positions = torch.arange(paths.shape[1], device=board.device).view(1, 1, -1)
        absent = torch.full_like(positions, paths.shape[1])
        first_position = torch.where(occupied, positions, absent).amin(dim=-1)
        first_exists = first_position.lt(paths.shape[1])
        second_occupied = occupied & positions.gt(first_position.unsqueeze(-1))
        second_position = torch.where(second_occupied, positions, absent).amin(dim=-1)
        second_exists = second_position.lt(paths.shape[1])

        expanded_paths = safe_paths.unsqueeze(0).expand(board.shape[0], -1, -1)
        first_squares = expanded_paths.gather(
            2, first_position.clamp_max(paths.shape[1] - 1).unsqueeze(-1)
        ).squeeze(-1)
        second_squares = expanded_paths.gather(
            2, second_position.clamp_max(paths.shape[1] - 1).unsqueeze(-1)
        ).squeeze(-1)
        source = hidden.index_select(1, self._ray_sources)
        target = hidden.index_select(1, self._ray_destinations)
        first = hidden.gather(
            1, first_squares.unsqueeze(-1).expand(-1, -1, self.dim)
        )
        second = hidden.gather(
            1, second_squares.unsqueeze(-1).expand(-1, -1, self.dim)
        )
        first = first * first_exists.unsqueeze(-1)
        second = second * second_exists.unsqueeze(-1)
        blocker_count = occupied.sum(dim=-1).clamp_max(3)

        values = (
            self.ray_message(source)
            + self.ray_target(target)
            + self.ray_first_blocker(first)
            + self.ray_second_blocker(second)
            + self.ray_direction(self._ray_directions).unsqueeze(0)
            + self.ray_distance(self._ray_distances).unsqueeze(0)
            + self.ray_blocker_count(blocker_count)
        )
        orthogonal_relation = self._ray_directions < len(_ORTHOGONAL_DIRECTIONS)
        source_active = torch.where(
            orthogonal_relation.unsqueeze(0),
            orthogonal_mask.index_select(1, self._ray_sources),
            diagonal_mask.index_select(1, self._ray_sources),
        )
        return values * source_active.unsqueeze(-1)

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
            edge_values = edge_values + self.ray_first_blocker(
                hidden.index_select(1, middles)
            )
            messages.index_add_(1, edge_destinations, edge_values)

        messages.index_add_(
            1,
            self._ray_destinations,
            self._ray_values(hidden, board, orthogonal_mask, diagonal_mask),
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
