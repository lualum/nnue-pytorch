import pytest
import torch

from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.features import FullThreats, PP3Wide
from model.modules.features.halfka_v2_hm import HalfKav2Hm, KingBuckets
from model.modules.movement import (
    OUR_KING,
    OUR_KNIGHT,
    OUR_PAWN,
    OUR_ROOK,
    THEIR_KING,
    MovementEvaluationNetwork,
    MovementFeatureDecoder,
)
from model.utils.movement_serialize import MOVEMENT_MAGIC, MovementNNUEWriter
from model.utils.serialize import NNUEWriter


def _zero_linear(layer: torch.nn.Linear) -> None:
    with torch.no_grad():
        layer.weight.zero_()
        if layer.bias is not None:
            layer.bias.zero_()


def _set_identity(layer: torch.nn.Linear) -> None:
    _zero_linear(layer)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(layer.out_features, layer.in_features))


def _halfka_index(piece_code: int, square: int, king_square: int = 4) -> int:
    bucket = KingBuckets[king_square]
    return bucket * HalfKav2Hm.NUM_PLANES + (piece_code - 1) * 64 + square


def test_decoder_uses_composed_offset_and_side_to_move_perspective():
    decoder = MovementFeatureDecoder("Full_Threats+PP_3Wide+HalfKAv2_hm^")
    halfka_offset = FullThreats.NUM_INPUTS + PP3Wide.NUM_INPUTS

    white_indices = torch.tensor(
        [[
            17,
            halfka_offset + _halfka_index(OUR_KING, 4),
            halfka_offset + _halfka_index(OUR_ROOK, 0),
            -1,
        ]],
        dtype=torch.int32,
    )
    black_indices = torch.tensor(
        [[
            halfka_offset + _halfka_index(OUR_KING, 4),
            halfka_offset + _halfka_index(OUR_KNIGHT, 18),
            -1,
            -1,
        ]],
        dtype=torch.int32,
    )

    white_board = decoder.decode(torch.ones(1, 1), white_indices, black_indices)
    black_board = decoder.decode(torch.zeros(1, 1), white_indices, black_indices)

    assert white_board[0, 0] == OUR_ROOK
    assert white_board[0, 4] == OUR_KING
    assert white_board[0, 18] == 0
    assert black_board[0, 18] == OUR_KNIGHT
    assert black_board[0, 0] == 0


def test_multiple_piece_messages_are_added():
    network = MovementEvaluationNetwork(dim=4, iterations=3)
    for layer in (
        network.knight_message,
        network.king_message,
        network.pawn_diagonal_message,
        network.pawn_forward_message,
        network.ray_message,
    ):
        _zero_linear(layer)
    _set_identity(network.knight_message)

    # Knights on b1 and f1 both cover d2.
    board = torch.zeros(1, 64, dtype=torch.long)
    board[0, 1] = OUR_KNIGHT
    board[0, 5] = OUR_KNIGHT
    hidden = torch.ones(1, 64, 4)
    messages = network.aggregate_messages(hidden, board)

    torch.testing.assert_close(messages[0, 11], torch.full((4,), 2.0))


def test_ray_continuation_is_learned_from_square_state_not_occupancy():
    network = MovementEvaluationNetwork(dim=4, iterations=3)
    for layer in (
        network.knight_message,
        network.king_message,
        network.pawn_diagonal_message,
        network.pawn_forward_message,
        network.ray_message,
        network.path_gate,
    ):
        _zero_linear(layer)
    _set_identity(network.ray_message)
    _set_identity(network.path_gate)
    with torch.no_grad():
        network.path_scale.zero_()

    board = torch.zeros(1, 64, dtype=torch.long)
    board[0, 0] = OUR_ROOK
    board[0, 16] = OUR_PAWN
    hidden = torch.full((1, 64, 4), -2.5)
    hidden[0, 0] = 1.0

    transmitting = network.aggregate_messages(hidden, board)
    torch.testing.assert_close(transmitting[0, 8], torch.ones(4))
    torch.testing.assert_close(transmitting[0, 16], torch.ones(4))
    torch.testing.assert_close(transmitting[0, 40], torch.ones(4))

    suppressing_hidden = hidden.clone()
    suppressing_hidden[0, 16] = 2.5
    suppressing = network.aggregate_messages(suppressing_hidden, board)
    torch.testing.assert_close(suppressing[0, 16], torch.ones(4))
    torch.testing.assert_close(suppressing[0, 24], torch.zeros(4))

    empty_board = board.clone()
    empty_board[0, 16] = 0
    empty_but_suppressing = network.aggregate_messages(suppressing_hidden, empty_board)
    torch.testing.assert_close(empty_but_suppressing[0, 24], torch.zeros(4))


def test_network_is_small_shared_weight_and_differentiable():
    network = MovementEvaluationNetwork(dim=16, iterations=4)
    board = torch.zeros(2, 64, dtype=torch.long)
    board[:, 4] = OUR_KING
    board[:, 60] = THEIR_KING
    board[0, 1] = OUR_KNIGHT
    board[1, 8] = OUR_PAWN

    result = network(board)
    assert result.shape == (2, 1)
    assert network.parameter_count < 10_000
    assert MovementEvaluationNetwork().parameter_count < 1_500
    assert sum(1 for name, _ in network.named_modules() if name == "update") == 1

    result.sum().backward()
    assert all(parameter.grad is not None for parameter in network.parameters())


def test_incremental_cache_matches_full_recalculation():
    torch.manual_seed(7)
    network = MovementEvaluationNetwork(dim=8, iterations=4)
    board = torch.zeros(64, dtype=torch.long)
    board[4] = OUR_KING
    board[60] = THEIR_KING
    board[0] = OUR_ROOK
    board[18] = OUR_KNIGHT
    board[16] = OUR_PAWN

    accumulator = network.create_accumulator(board)
    moved = board.clone()
    moved[0] = 0
    moved[8] = OUR_ROOK
    updated = network.update_accumulator(accumulator, moved)
    full = network(moved.unsqueeze(0))

    torch.testing.assert_close(updated.evaluation, full, rtol=1e-5, atol=1e-6)
    assert updated.updated_squares[0] == 2
    assert all(
        before <= after
        for before, after in zip(
            updated.updated_squares, updated.updated_squares[1:]
        )
    )


def test_model_training_interface_and_legacy_export_guard():
    config = ModelConfig(
        network_type="movement", movement_dim=8, movement_iterations=3
    )
    model = NNUEModel("HalfKAv2_hm^", config)
    indices = torch.tensor(
        [[
            _halfka_index(OUR_KING, 4),
            _halfka_index(THEIR_KING, 60),
            _halfka_index(OUR_KNIGHT, 1),
            -1,
        ]],
        dtype=torch.int32,
    )
    result = model(
        torch.ones(1, 1),
        torch.zeros(1, 1),
        indices,
        indices,
        torch.tensor([3]),
        False,
        False,
    )
    assert result.shape == (1, 1)

    with pytest.raises(ValueError, match="cannot encode movement graphs"):
        NNUEWriter(model, verbose=False)

    movement_writer = MovementNNUEWriter(model, "test movement net")
    assert movement_writer.buf.startswith(MOVEMENT_MAGIC)
    assert b"piece_embedding.weight" in movement_writer.buf
    assert b"path_gate.weight" in movement_writer.buf
    assert b"xray_gate" not in movement_writer.buf
