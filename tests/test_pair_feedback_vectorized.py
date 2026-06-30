import pytest

torch = pytest.importorskip("torch")

from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.chemistry.graphs import Vocab
from kgcl_retro.models import KGCL


def _make_contextual_model():
    atom_vocab = Vocab([("Change Atom", (0, 0))])
    bond_vocab = Vocab([("Add Bond", (1.0, None))])
    config = {
        "model_variant": "contextual_2fwl",
        "n_atom_feat": ATOM_FDIM,
        "n_bond_feat": ATOM_FDIM + BOND_FDIM,
        "mpn_size": 8,
        "mlp_size": 16,
        "depth": 1,
        "dropout_mlp": 0.0,
        "dropout_mpn": 0.0,
        "atom_message": False,
        "use_attn": False,
        "n_heads": 1,
        "fg_hidden_size": 8,
        "pair_hidden_size": 8,
        "pair_relation_size": 4,
        "pair_enc_layers": 1,
        "pair_dec_layers": 1,
        "pair_topk": 8,
    }
    return KGCL(config=config, atom_vocab=atom_vocab, bond_vocab=bond_vocab).contextual_2fwl


def test_vectorized_pair_feedback_matches_reference():
    model = _make_contextual_model()
    torch.manual_seed(11)
    atom_states = torch.randn((5, 8))
    pair_states = torch.randn((4, 8))
    pair_rel = torch.randn((4, 4))
    pairs = torch.tensor([[1, 1], [1, 2], [2, 1], [3, 2]], dtype=torch.long)

    expected = model._apply_pair_feedback_reference(atom_states, pair_states, pair_rel, pairs)
    actual = model._apply_pair_feedback(atom_states, pair_states, pair_rel, pairs)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_pair_feedback_empty_pairs_returns_atom_states():
    model = _make_contextual_model()
    atom_states = torch.randn((5, 8))

    actual = model._apply_pair_feedback(
        atom_states,
        torch.zeros((0, 8)),
        torch.zeros((0, 4)),
        torch.zeros((0, 2), dtype=torch.long),
    )

    assert torch.equal(actual, atom_states)


def test_pair_feedback_backward_pass():
    model = _make_contextual_model()
    atom_states = torch.randn((5, 8), requires_grad=True)
    pair_states = torch.randn((4, 8), requires_grad=True)
    pair_rel = torch.randn((4, 4), requires_grad=True)
    pairs = torch.tensor([[1, 1], [1, 2], [2, 1], [3, 2]], dtype=torch.long)

    loss = model._apply_pair_feedback(atom_states, pair_states, pair_rel, pairs).sum()
    loss.backward()

    assert atom_states.grad is not None
    assert pair_states.grad is not None
    assert pair_rel.grad is not None


def test_padding_atom_not_corrupted_if_padding_present():
    model = _make_contextual_model()
    atom_states = torch.randn((4, 8))
    atom_states[0].zero_()
    pair_states = torch.randn((2, 8))
    pair_rel = torch.randn((2, 4))
    pairs = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    actual = model._apply_pair_feedback(atom_states, pair_states, pair_rel, pairs)

    assert torch.allclose(actual[0], torch.zeros_like(actual[0]))
