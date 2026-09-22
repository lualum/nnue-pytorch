import io

import pytest
import torch

from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.movement import (
    OUR_PAWN,
    OUR_ROOK,
    MovementEvaluationNetwork,
)
from model.utils.movement_serialize import MovementNNUEWriter


def test_ray_pairs_cover_xrays_with_ordered_blocker_slots():
    network = MovementEvaluationNetwork(dim=4)
    board = torch.zeros(1, 64, dtype=torch.long)
    board[0, 0] = OUR_ROOK
    board[0, 16] = OUR_PAWN
    board[0, 32] = OUR_PAWN
    hidden = torch.zeros(1, 64, 4)
    hidden[0, 0] = 1.0
    hidden[0, 16] = 2.0
    hidden[0, 32] = 3.0
    for layer in (network.ray_target, network.ray_first_blocker, network.ray_second_blocker):
        with torch.no_grad():
            layer.weight.copy_(torch.eye(4))
    with torch.no_grad():
        network.ray_message.weight.copy_(torch.eye(4))
        network.ray_message.bias.zero_()
        network.ray_direction.weight.zero_()
        network.ray_distance.weight.zero_()
        network.ray_blocker_count.weight.zero_()
    values = network._ray_values(
        hidden, board, board == OUR_ROOK, torch.zeros_like(board, dtype=torch.bool)
    )
    relation = (network._ray_sources == 0) & (network._ray_destinations == 48)
    # a1 -> a7 gets source, first blocker a3, and second blocker a5 directly.
    torch.testing.assert_close(values[0, relation][0], torch.full((4,), 6.0))


def test_ray_blocker_changes_are_visible_without_recurrent_path_state():
    network = MovementEvaluationNetwork(dim=8)
    board = torch.zeros(1, 64, dtype=torch.long)
    board[0, 0] = OUR_ROOK
    hidden = torch.randn(1, 64, 8)
    direct = network._ray_values(
        hidden, board, board == OUR_ROOK, torch.zeros_like(board, dtype=torch.bool)
    )
    board[0, 16] = OUR_PAWN
    blocked = network._ray_values(
        hidden, board, board == OUR_ROOK, torch.zeros_like(board, dtype=torch.bool)
    )
    relation = (network._ray_sources == 0) & (network._ray_destinations == 40)
    assert not torch.allclose(direct[0, relation], blocked[0, relation])


def test_direct_ray_cache_matches_full_evaluation_after_multiple_edits():
    torch.manual_seed(23)
    network = MovementEvaluationNetwork()
    board = torch.zeros(64, dtype=torch.long)
    board[[4, 60, 0, 16, 32]] = torch.tensor([11, 12, 7, 1, 10])
    cache = network.create_accumulator(board)
    for edits in ({16: 0, 24: 1}, {0: 0, 32: 7}, {24: 0, 56: 9}, {4: 0, 6: 11, 7: 0, 5: 7}):
        board = board.clone()
        for square, code in edits.items():
            board[square] = code
        cache = network.update_accumulator(cache, board)
        torch.testing.assert_close(cache.evaluation, network(board[None]))


def test_training_interface_checkpoint_roundtrip_and_export_guard():
    torch.manual_seed(29)
    config = ModelConfig(network_type="movement")
    model = NNUEModel("HalfKAv2_hm^", config)
    indices = torch.tensor([[640 + 4, 704 + 60, 384, -1]], dtype=torch.int32)
    inputs = (torch.ones(1, 1), torch.zeros(1, 1), indices, indices,
              torch.tensor([3]), False, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial = model(*inputs).detach()
    target = initial + 0.5
    for _ in range(8):
        optimizer.zero_grad()
        loss = (model(*inputs) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert (model(*inputs) - target).square().item() < 0.25
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = NNUEModel("HalfKAv2_hm^", config)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(*inputs), model(*inputs))
    with pytest.raises(ValueError, match="Direct ray relations"):
        MovementNNUEWriter(model)
