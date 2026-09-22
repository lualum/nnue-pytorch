import io

import pytest
import torch

from model.config import ModelConfig
from model.model import NNUEModel
from model.modules.movement import MovementEvaluationNetwork
from model.utils.movement_serialize import MovementNNUEWriter


def test_zero_mixer_matches_baseline_and_receives_gradients():
    torch.manual_seed(17)
    baseline = MovementEvaluationNetwork()
    ordered = MovementEvaluationNetwork(ordered_rays=True)
    missing = ordered.load_state_dict(baseline.state_dict(), strict=False)
    assert missing.missing_keys == ["path_mix"]
    assert not missing.unexpected_keys
    board = torch.randint(0, 13, (2, 64))
    torch.testing.assert_close(ordered(board), baseline(board), rtol=0, atol=0)
    ordered(board).sum().backward()
    assert ordered.path_mix.grad.abs().sum() > 0


def test_intervening_state_order_is_distinguishable():
    network = MovementEvaluationNetwork(dim=4, ordered_rays=True)
    with torch.no_grad():
        network.path_mix.copy_(torch.eye(4).roll(1, dims=1))
    signal = torch.tensor([1., 2., 3., 4.])
    first = torch.tensor([0.1, 0.3, 0.5, 0.7])
    second = first.flip(0)
    transition = network._path_transition
    forward = transition(transition(signal, first), second)
    reverse = transition(transition(signal, second), first)
    assert not torch.allclose(forward, reverse)
    network.ordered_rays = False
    torch.testing.assert_close(
        transition(transition(signal, first), second),
        transition(transition(signal, second), first),
    )


def test_transition_preserves_additivity_and_bounds_signal():
    torch.manual_seed(19)
    network = MovementEvaluationNetwork(ordered_rays=True)
    with torch.no_grad():
        network.path_mix.normal_(std=5)
    a, b = torch.randn(2, 100, 8)
    factors = torch.rand(100, 8) * 2 - 1
    transition = network._path_transition
    torch.testing.assert_close(
        transition(a + b, factors), transition(a, factors) + transition(b, factors)
    )
    assert torch.all(transition(a, factors).abs().amax(-1) <= a.abs().amax(-1) + 1e-6)


def test_ordered_cache_matches_full_evaluation_after_multiple_edits():
    torch.manual_seed(23)
    network = MovementEvaluationNetwork(ordered_rays=True)
    with torch.no_grad():
        network.path_mix.normal_(std=0.2)
    board = torch.zeros(64, dtype=torch.long)
    board[[4, 60, 0, 16, 32]] = torch.tensor([11, 12, 7, 1, 10])
    cache = network.create_accumulator(board)
    # Captures, removing a blocker, promotion, and a multi-square edit exercise
    # dependency invalidation. These are structural tests, not legal games.
    for edits in ({16: 0, 24: 1}, {0: 0, 32: 7}, {24: 0, 56: 9}, {4: 0, 6: 11, 7: 0, 5: 7}):
        board = board.clone()
        for square, code in edits.items():
            board[square] = code
        cache = network.update_accumulator(cache, board)
        torch.testing.assert_close(cache.evaluation, network(board[None]))
    unchanged = network.update_accumulator(cache, board)
    assert not any(unchanged.updated_squares)


def test_training_interface_checkpoint_roundtrip_and_export_guard():
    torch.manual_seed(29)
    config = ModelConfig(network_type="movement", movement_ordered_rays=True)
    model = NNUEModel("HalfKAv2_hm^", config)
    # Bucket zero: piece-plane offset plus square. Sparse input uses the
    # existing trainer interface, including its side-to-move selection.
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
    assert model.movement.path_mix.abs().sum() > 0
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = NNUEModel("HalfKAv2_hm^", config)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(*inputs), model(*inputs))
    with pytest.raises(ValueError, match="matching native runtime"):
        MovementNNUEWriter(model)
