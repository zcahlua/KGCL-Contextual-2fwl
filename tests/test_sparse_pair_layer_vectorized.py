import inspect

import pytest

torch = pytest.importorskip("torch")

from kgcl_retro.models.sparse_pair import SparsePairLayerReference, SparsePairLayerVectorized


def _make_layer_pair():
    torch.manual_seed(7)
    reference = SparsePairLayerReference(pair_hidden_size=5, pair_relation_size=3)
    vectorized = SparsePairLayerVectorized(pair_hidden_size=5, pair_relation_size=3)
    vectorized.load_state_dict(reference.state_dict())
    return reference, vectorized


def test_vectorized_matches_reference_tiny_chain():
    reference, vectorized = _make_layer_pair()
    pairs = torch.tensor([[1, 1], [1, 2], [2, 1], [2, 2], [2, 3], [3, 2], [3, 3]], dtype=torch.long)
    bridge_index = torch.tensor([[0, 0], [1, 2], [2, 1], [0, 0], [2, 3], [3, 2], [0, 0]], dtype=torch.long)
    bridge_mask = torch.tensor([[False, False], [True, True], [True, True], [False, False], [True, True], [True, True], [False, False]])
    pair_states = torch.randn((pairs.size(0), 5))
    pair_rel = torch.randn((pairs.size(0), 3))

    expected = reference(pair_states, pair_rel, pairs, bridge_index, bridge_mask)
    actual = vectorized(pair_states, pair_rel, pairs, bridge_index, bridge_mask)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_vectorized_matches_reference_ring():
    reference, vectorized = _make_layer_pair()
    pairs = torch.tensor(
        [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]],
        dtype=torch.long,
    )
    bridge_index = torch.tensor(
        [[0, 0, 0], [1, 2, 3], [1, 2, 3], [2, 1, 3], [0, 0, 0], [2, 3, 1], [3, 1, 2], [3, 2, 1], [0, 0, 0]],
        dtype=torch.long,
    )
    bridge_mask = bridge_index > 0
    pair_states = torch.randn((pairs.size(0), 5))
    pair_rel = torch.randn((pairs.size(0), 3))

    expected = reference(pair_states, pair_rel, pairs, bridge_index, bridge_mask)
    actual = vectorized(pair_states, pair_rel, pairs, bridge_index, bridge_mask)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_vectorized_handles_empty_pairs():
    _reference, vectorized = _make_layer_pair()
    pair_states = torch.zeros((0, 5))
    pair_rel = torch.zeros((0, 3))
    pairs = torch.zeros((0, 2), dtype=torch.long)
    bridge_index = torch.zeros((0, 2), dtype=torch.long)
    bridge_mask = torch.zeros((0, 2), dtype=torch.bool)

    output = vectorized(pair_states, pair_rel, pairs, bridge_index, bridge_mask)

    assert output.shape == (0, 5)


def test_vectorized_handles_empty_bridges():
    reference, vectorized = _make_layer_pair()
    pairs = torch.tensor([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=torch.long)
    bridge_index = torch.zeros((4, 2), dtype=torch.long)
    bridge_mask = torch.zeros((4, 2), dtype=torch.bool)
    pair_states = torch.randn((pairs.size(0), 5))
    pair_rel = torch.randn((pairs.size(0), 3))

    expected = reference(pair_states, pair_rel, pairs, bridge_index, bridge_mask)
    actual = vectorized(pair_states, pair_rel, pairs, bridge_index, bridge_mask)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_vectorized_backward_pass():
    _reference, vectorized = _make_layer_pair()
    pairs = torch.tensor([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=torch.long)
    bridge_index = torch.tensor([[0, 0], [1, 2], [2, 1], [0, 0]], dtype=torch.long)
    bridge_mask = bridge_index > 0
    pair_states = torch.randn((pairs.size(0), 5), requires_grad=True)
    pair_rel = torch.randn((pairs.size(0), 3), requires_grad=True)

    loss = vectorized(pair_states, pair_rel, pairs, bridge_index, bridge_mask).sum()
    loss.backward()

    assert pair_states.grad is not None
    assert pair_rel.grad is not None


def test_vectorized_no_cpu_tolist_hotpath_if_reasonable_to_check():
    source = inspect.getsource(SparsePairLayerVectorized.forward)

    assert ".tolist()" not in source
