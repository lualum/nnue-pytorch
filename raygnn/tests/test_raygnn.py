import random
from dataclasses import replace

import chess
import pytest
import torch

from raygnn import RayGNN, RayGNNConfig, RayGNNEvaluator, boards_to_batch, fens_to_batch
from raygnn.encoding import swap_colors_reflect_ranks
from raygnn.geometry import BoardGeometry
from raygnn.train import TeacherDataset


def test_reference_shapes_gradients_and_forward_contract():
    batch = boards_to_batch([chess.Board(), chess.Board()])
    model = RayGNN()
    assert len(model.layers) == 2
    assert model.layers[0] is not model.layers[1]
    assert model.layers[0].update[0].in_features == 310
    assert model.readout.spatial.in_features == 192
    assert model.readout.head[0].in_features == 528
    assert model.encoder(batch.piece).shape == (2, 64, 96)
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
    for weight in (model.encoder.joint_embedding.weight, model.edge_encoder.weight,
                   model.layers[0].source_project.weight, model.layers[1].source_project.weight,
                   model.readout.head[0].weight):
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


def test_fast_fen_encoding_matches_board_encoding():
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    ]
    fast = fens_to_batch(fens)
    reference = boards_to_batch([chess.Board(fen) for fen in fens])
    for field in ("piece", "side_to_move", "castling", "en_passant"):
        assert torch.equal(getattr(fast, field), getattr(reference, field))


def test_draw_state_contract_requires_history_in_training_records():
    row = {"fen": chess.STARTING_FEN, "eval_cp": 50}
    with pytest.raises(ValueError, match="repetition_twofold"):
        TeacherDataset([row], "cp_pawns", use_draw_state=True)
    dataset = TeacherDataset([{**row, "repetition_twofold": True}], "cp_pawns", use_draw_state=True)
    assert dataset[0][1] == 0.5
    batch = boards_to_batch([chess.Board()])
    model = RayGNN(replace(RayGNNConfig.variant("no_message"),
                           draw_state_fields=("halfmove_clock_div_100", "repetition_twofold")))
    assert model(batch).shape == (1, 1)
    with pytest.raises(ValueError, match="draw_state"):
        model(batch.piece, batch.side_to_move, batch.castling, batch.en_passant)


@pytest.mark.parametrize("variant", [
    "reference", "no_message", "no_relation_separation", "dense_context",
    "king_relative", "wider_spatial", "wider_messages", "global_pooling",
    "three_layers", "simple_pair",
])
def test_variants_forward(variant):
    batch = boards_to_batch([chess.Board()])
    model = RayGNN(RayGNNConfig.variant(variant))
    output = model(batch)
    assert output.shape == (1, 1) and torch.isfinite(output).all()
    if variant == "no_relation_separation":
        assert model.layers[0].update[0].in_features == 178
    if variant == "wider_messages":
        assert model.layers[0].update[0].in_features == 406


def test_dense_reference_and_chunked_outputs_and_gradients():
    batch = boards_to_batch([chess.Board()])
    reference = RayGNN()
    for change in (dict(edge_chunk_size=137), dict(dense_reference=True)):
        variant = RayGNN(replace(reference.config, **change))
        variant.load_state_dict(reference.state_dict())
        first, second = reference(batch), variant(batch)
        assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)
        first.sum().backward()
        second.sum().backward()
        for name in ("encoder.joint_embedding.weight", "edge_encoder.weight",
                     "layers.0.pair.0.0.weight", "layers.0.pair.1.0.weight",
                     "layers.0.pair.2.0.weight", "readout.head.0.weight"):
            first_grad = dict(reference.named_parameters())[name].grad
            second_grad = dict(variant.named_parameters())[name].grad
            assert first_grad is not None and second_grad is not None
            assert torch.allclose(first_grad, second_grad, atol=1e-5, rtol=1e-5), name
        reference.zero_grad()


def test_relationship_precedence_and_inactive_edges():
    geometry = BoardGeometry()
    piece = torch.zeros((1, 64), dtype=torch.long)
    piece[0, chess.A1] = 4  # rook
    piece[0, chess.A3] = 1  # first blocker
    piece[0, chess.A5] = 1  # second blocker
    _, relation = geometry.edge_features(piece)
    def edge(src, dst):
        return ((geometry.src == src) & (geometry.dst == dst)).nonzero()[0, 0]
    assert relation[0, edge(chess.A1, chess.A2)] == 0
    assert relation[0, edge(chess.A1, chess.A4)] == 1
    assert relation[0, edge(chess.A1, chess.A6)] == -1
    assert relation[0, edge(chess.B2, chess.C3)] == 2  # empty-source adjacency
    assert relation[0, edge(chess.B2, chess.E5)] == -1



def test_engine_scores_and_validation():
    board = chess.Board()
    batch = boards_to_batch([board])
    model = RayGNN(RayGNNConfig.variant("no_message"))
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
