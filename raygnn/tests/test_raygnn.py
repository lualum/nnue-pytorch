import random

import chess
import pytest
import torch

from raygnn import RayGNN, RayGNNConfig, RayGNNEvaluator, boards_to_batch
from raygnn.encoding import swap_colors_rotate
from raygnn.geometry import BoardGeometry


def test_starting_position_shapes_and_gradients():
    batch = boards_to_batch([chess.Board(), chess.Board()])
    model = RayGNN(RayGNNConfig(wdl_head=True))
    assert model.head[0].in_features == 928
    assert model.layers[0].update[0].in_features == 224
    encoded = model.encoder(batch)
    assert encoded.shape == (2, 64, 64)
    rays = model.geometry.rays(batch.piece)
    assert rays.valid.shape == (2, 64, 8, 7)
    assert model.readout(encoded, batch.piece, model.geometry).shape == (2, 832)
    output = model(batch)
    assert output.value.shape == (2, 1)
    assert output.wdl_logits.shape == (2, 3)
    assert torch.isfinite(output.value).all()
    assert torch.allclose(output.material, torch.zeros_like(output.material))
    output.value.sum().backward()
    assert model.encoder.project.weight.grad is not None
    assert torch.isfinite(model.encoder.project.weight.grad).all()


def test_empty_piece_groups_and_ablations():
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    batch = boards_to_batch([board])
    for config in (
        RayGNNConfig(layers=0, structured_readout=False, raw_board_skip=False),
        RayGNNConfig(layers=1, use_xray=False),
        RayGNNConfig(layers=1, first_piece_only=True),
        RayGNNConfig(layers=4),
    ):
        result = RayGNN(config)(batch)
        assert result.value.shape == (1, 1)
        assert torch.isfinite(result.value).all()


def test_engine_api_is_white_positive_centipawns():
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    model = RayGNN(RayGNNConfig(layers=0))
    evaluator = RayGNNEvaluator(model)
    expected = model(boards_to_batch([board])).value[:, 0] * 100
    assert torch.allclose(evaluator.evaluate_cp([board]), expected)


def test_board_state_encoding_and_color_swap():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("a6")
    board.push_san("e5")
    board.push_san("d5")
    batch = boards_to_batch([board])
    assert batch.en_passant.item() == chess.D6
    swapped = swap_colors_rotate(batch)
    assert torch.equal(swap_colors_rotate(swapped).piece, batch.piece)
    assert torch.equal(swap_colors_rotate(swapped).castling, batch.castling)
    assert torch.equal(swap_colors_rotate(swapped).en_passant, batch.en_passant)
    model = RayGNN()
    assert torch.allclose(model(batch).material, -model(swapped).material)
    assert torch.allclose(model(batch).value, -model(swapped).value, atol=1e-5)


def test_ray_relations_against_python_chess():
    rng = random.Random(23)
    boards = [chess.Board()]
    for _ in range(8):
        board = chess.Board()
        for _ in range(rng.randrange(4, 50)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        boards.append(board)
    batch = boards_to_batch(boards)
    geometry = BoardGeometry()
    relations = geometry.rays(batch.piece)
    for row, board in enumerate(boards):
        for destination in range(64):
            for direction in range(8):
                for distance in range(7):
                    source = geometry.ray_index[destination, direction, distance].item()
                    if source < 0:
                        assert not relations.valid[row, destination, direction, distance]
                        continue
                    # Ordered ray slots between destination and source, excluding both endpoints.
                    interior = [geometry.ray_index[destination, direction, k].item() for k in range(distance)]
                    blockers = [sq for sq in interior if board.piece_at(sq)]
                    expected_direct = not blockers
                    expected_source = board.piece_at(source)
                    expected_attack = expected_source is not None and destination in board.attacks(source)
                    assert relations.direct[row, destination, direction, distance].item() == expected_direct
                    assert relations.xray[row, destination, direction, distance].item() == (not expected_direct)
                    assert relations.attack[row, destination, direction, distance].item() == expected_attack
                    assert relations.blocker_count[row, destination, direction, distance].item() == min(len(blockers), 2)
                    first = board.piece_at(blockers[-1]) if blockers else None
                    expected_id = first.piece_type + (0 if first.color else 6) if first else 0
                    assert relations.blocker_piece[row, destination, direction, distance].item() == expected_id
                    assert relations.blocker_square[row, destination, direction, distance].item() == (blockers[-1] if blockers else -1)


@pytest.mark.parametrize("fen", [
    "4k3/8/8/3p4/3R4/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/3P4/4K3 w - - 0 1",
    "4k3/8/8/8/3n4/8/8/4K3 b - - 0 1",
])
def test_jump_sources_and_xray_cases(fen):
    board = chess.Board(fen)
    batch = boards_to_batch([board])
    geometry = BoardGeometry()
    for relation in ("knight", "pawn"):
        indices, _, mask = geometry.jump_sources(batch.piece, relation)
        for destination in range(64):
            for slot in range(indices.shape[-1]):
                if not mask[0, destination, slot]:
                    continue
                source = indices[destination, slot].item()
                piece = board.piece_at(source)
                assert piece is not None
                assert piece.piece_type == (chess.KNIGHT if relation == "knight" else chess.PAWN)
                assert destination in board.attacks(source)
