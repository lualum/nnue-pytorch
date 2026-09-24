import random
from dataclasses import replace

import chess
import pytest
import torch

from raygnn import RayGNN, RayGNNConfig, RayGNNEvaluator, boards_to_batch
from raygnn.encoding import swap_colors_reflect_ranks
from raygnn.geometry import BoardGeometry
from raygnn.train import TeacherDataset


def test_reference_shapes_gradients_and_forward_contract():
    batch = boards_to_batch([chess.Board(), chess.Board()])
    model = RayGNN()
    assert len(model.layers) == 2
    assert model.layers[0] is not model.layers[1]
    assert model.layers[0].update[0].in_features == 144
    assert model.readout.first.in_features == 128
    assert model.head[0].in_features == 976
    assert model.encoder(batch.piece).shape == (2, 64, 64)
    assert model.state_encoder(batch.side_to_move, batch.castling, batch.en_passant).shape == (2, 16)
    features, classes = model.geometry.edge_features(batch.piece)
    assert features.shape == (2, 1792, 31)
    assert classes.shape == (2, 1792)
    assert model.edge_encoder(features).shape == (2, 1792, 32)
    assert torch.allclose(model.layers[0].alpha, torch.tensor(0.1))
    value = model(batch.piece, batch.side_to_move, batch.castling, batch.en_passant)
    assert value.shape == (2, 1)
    assert torch.allclose(value, model(batch))
    value.sum().backward()
    for weight in (model.encoder.project.weight, model.edge_encoder.weight,
                   model.layers[0].source_project.weight, model.layers[1].source_project.weight,
                   model.head[0].weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_static_edges_and_features_against_chess():
    geometry = BoardGeometry()
    assert geometry.src.shape == geometry.dst.shape == (1792,)
    assert geometry.between.shape == (1792, 6)
    assert geometry.between_mask.shape == (1792, 6)
    assert geometry.relative_delta.shape == (1792, 2)
    assert geometry.is_knight.sum().item() == 336
    assert not (geometry.src == geometry.dst).any()
    pairs = set(zip(geometry.src.tolist(), geometry.dst.tolist()))
    assert len(pairs) == 1792
    assert all((dst, src) in pairs for src, dst in pairs)
    rng = random.Random(23)
    boards = [chess.Board()]
    for _ in range(5):
        board = chess.Board()
        for _ in range(rng.randrange(4, 50)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        boards.append(board)
    features, _ = geometry.edge_features(boards_to_batch(boards).piece)
    for row, board in enumerate(boards):
        for edge, (src, dst) in enumerate(zip(geometry.src.tolist(), geometry.dst.tolist())):
            df, dr = dst % 8 - src % 8, dst // 8 - src // 8
            if geometry.is_knight[edge]:
                between = []
            else:
                sf, sr = (df > 0) - (df < 0), (dr > 0) - (dr < 0)
                between = [src + k * (8 * sr + sf) for k in range(1, max(abs(df), abs(dr)))]
            blockers = [sq for sq in between if board.piece_at(sq)]
            piece = board.piece_at(src)
            expected_attack = piece is not None and dst in board.attacks(src)
            vector = features[row, edge]
            assert vector[3:10].argmax().item() == len(blockers)
            first = board.piece_at(blockers[0]) if blockers else None
            first_id = first.piece_type + (0 if first.color else 6) if first else 0
            assert vector[10:23].argmax().item() == first_id
            assert vector[25].item() == bool(blockers)  # blocker present
            assert vector[26].item() == expected_attack or (piece is not None and vector[26].item() and bool(blockers))
            assert vector[27].item() == expected_attack
            assert vector[28].item() == (piece is not None)
            assert vector[29].item() == (board.piece_at(dst) is not None)
    # Pawn forward geometry is separate from attack geometry.
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    features, _ = geometry.edge_features(boards_to_batch([board]).piece)
    edge = ((geometry.src == chess.E2) & (geometry.dst == chess.E4)).nonzero()[0, 0]
    assert features[0, edge, 30] == 1
    assert features[0, edge, 27] == 0


def test_six_blockers_and_padded_slots():
    geometry = BoardGeometry()
    piece = torch.zeros((1, 64), dtype=torch.long)
    piece[0, chess.A1] = 4
    for square in (chess.A2, chess.A3, chess.A4, chess.A5, chess.A6, chess.A7):
        piece[0, square] = 1
    edge = ((geometry.src == chess.A1) & (geometry.dst == chess.A8)).nonzero()[0, 0]
    features, _ = geometry.edge_features(piece)
    assert features[0, edge, 3:10].argmax() == 6
    assert features[0, edge, 10:23].argmax() == 1
    assert torch.allclose(features[0, edge, 23:25], torch.tensor([0.0, 1 / 7]))
    assert features[0, edge, 26] == 1  # rook geometry
    assert features[0, edge, 27] == 0  # blocked attack
    empty_knight = ((geometry.src == chess.B1) & (geometry.dst == chess.A3)).nonzero()[0, 0]
    assert not geometry.between_mask[empty_knight].any()
    assert features[0, empty_knight, 25:28].sum() == 0


def test_state_en_passant_and_rank_reflection():
    board = chess.Board()
    board.push_san("e4")
    batch = boards_to_batch([board])
    assert batch.en_passant.item() == chess.E3  # FEN target, even without a legal capture
    transformed = swap_colors_reflect_ranks(batch)
    restored = swap_colors_reflect_ranks(transformed)
    for name in ("piece", "side_to_move", "castling", "en_passant", "draw_state"):
        assert torch.equal(getattr(restored, name), getattr(batch, name))
    assert transformed.en_passant.item() == chess.E6
    assert transformed.side_to_move.item() == 1
    assert transformed.castling.tolist() == [[1, 1, 1, 1]]


def test_draw_state_contract_requires_history_in_training_records():
    row = {"fen": chess.STARTING_FEN, "eval_cp": 50}
    with pytest.raises(ValueError, match="repetition_twofold"):
        TeacherDataset([row], "cp_pawns", use_draw_state=True)
    dataset = TeacherDataset([{**row, "repetition_twofold": True}], "cp_pawns", use_draw_state=True)
    assert dataset[0][1] == 0.5
    batch = boards_to_batch([chess.Board()])
    model = RayGNN(replace(RayGNNConfig.variant("raw_board_only"),
                           draw_state_fields=("halfmove_clock_div_100", "repetition_twofold")))
    assert model(batch).shape == (1, 1)
    with pytest.raises(ValueError, match="draw_state"):
        model(batch.piece, batch.side_to_move, batch.castling, batch.en_passant)


@pytest.mark.parametrize("variant", [
    "reference", "original_compact", "no_message", "raw_board_only", "channelwise_gates",
    "king_relative", "smaller_messages", "smaller_readout", "relation_biases",
    "one_layer", "three_layers",
])
def test_variants_forward(variant):
    batch = boards_to_batch([chess.Board()])
    model = RayGNN(RayGNNConfig.variant(variant))
    output = model(batch)
    assert output.shape == (1, 1) and torch.isfinite(output).all()
    if variant == "original_compact":
        assert model.layers[0].update[0].in_features == 112
        assert model.head[0].in_features == 912
    if variant == "raw_board_only":
        assert model.head[0].in_features == 848


def test_implementation_variants_match_reference_outputs_and_gradients():
    batch = boards_to_batch([chess.Board()])
    reference = RayGNN()
    for change in (dict(edge_chunk_size=137), dict(float32_reductions=True),
                   dict(sparse_board_projection=True)):
        variant = RayGNN(replace(reference.config, **change))
        variant.load_state_dict(reference.state_dict())
        first, second = reference(batch), variant(batch)
        assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)
        first.sum().backward()
        second.sum().backward()
        assert torch.allclose(reference.head[0].weight.grad,
                              variant.head[0].weight.grad, atol=1e-5, rtol=1e-5)
        reference.zero_grad()


def test_engine_scores_and_validation():
    board = chess.Board()
    batch = boards_to_batch([board])
    model = RayGNN(RayGNNConfig.variant("raw_board_only"))
    evaluator = RayGNNEvaluator(model)
    expected = torch.round(100 * model(batch)[:, 0]).long()
    assert torch.equal(evaluator.evaluate_cp([board]), expected)
    assert torch.equal(evaluator.evaluate_cp([board], side_to_move=True), expected)
    black = chess.Board()
    black.turn = chess.BLACK
    expected_black = torch.round(-100 * model(boards_to_batch([black]))[:, 0]).long()
    assert torch.equal(evaluator.evaluate_cp([black], side_to_move=True), expected_black)
    with pytest.raises(ValueError):
        model(batch.piece, batch.side_to_move, batch.castling, torch.tensor([-1]))
