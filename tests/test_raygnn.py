import torch

from model.modules.raygnn import (
    BLACK_KING,
    BLACK_ROOK,
    WHITE_KING,
    WHITE_PAWN,
    WHITE_ROOK,
    RayGNNEvaluationNetwork,
    RayGNNPosition,
)
from model.modules.raygnn_data import collate_binpack_raygnn


def test_fen_preserves_board_and_state_fields():
    position = RayGNNPosition.from_fens(
        "r3k2r/8/8/3pP3/8/8/8/R3K2R b KQkq d6 17 42", repetition_count=2
    )
    assert position.board.shape == (1, 64)
    assert position.board[0, 0] == WHITE_ROOK
    assert position.board[0, 4] == WHITE_KING
    assert position.board[0, 56] == BLACK_ROOK
    assert position.board[0, 60] == BLACK_KING
    assert position.side_to_move.item() == 0
    assert position.castling_rights.tolist() == [[1.0, 1.0, 1.0, 1.0]]
    assert position.en_passant.item() == 43  # d6
    assert position.halfmove_clock.item() == 17
    assert position.repetition_count.item() == 2


def test_ray_features_separate_direct_and_xray_relations():
    model = RayGNNEvaluationNetwork(layers=1)
    board = torch.zeros(1, 64, dtype=torch.long)
    board[0, 4] = WHITE_KING
    board[0, 60] = BLACK_KING
    board[0, 0] = WHITE_ROOK
    board[0, 16] = WHITE_PAWN
    edge, valid = model._ray_features(board)
    # a1 to a2 is direct (destination a2, source a1, north-going source edge).
    assert valid[0, 8, 2, 0]
    assert edge[0, 8, 2, 0, 1] == 1
    # a1 to a4 crosses the pawn on a3: x-ray, not direct.
    assert edge[0, 24, 2, 2, 1] == 0
    assert edge[0, 24, 2, 2, 2] == 1


def test_full_model_has_documented_shapes_and_gradients():
    model = RayGNNEvaluationNetwork(layers=3, with_wdl=True)
    position = RayGNNPosition.from_fens([
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 b - - 12 8",
    ], repetition_count=[0, 1])
    value, wdl = model(position, return_wdl=True)
    assert value.shape == (2, 1)
    assert wdl.shape == (2, 3)
    assert torch.isfinite(value).all()
    (value.square().mean() + wdl.square().mean()).backward()
    assert model.value_head[-1].weight.grad is not None


def test_material_baseline_is_explicit_when_correction_is_zeroed():
    model = RayGNNEvaluationNetwork(layers=1)
    for parameter in model.value_head.parameters():
        parameter.data.zero_()
    position = RayGNNPosition.from_fens("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    # White has a rook more, so the documented material component is +5 pawns.
    torch.testing.assert_close(model(position), torch.tensor([[5.0]]))


def test_binpack_side_to_move_scores_become_white_positive_pawns():
    batch = collate_binpack_raygnn([
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 125, 0),
        ("4k3/8/8/8/8/8/8/4K3 b - - 0 1", 125, 0),
    ])
    torch.testing.assert_close(
        batch["score_pawns_white"], torch.tensor([[1.25], [-1.25]])
    )
