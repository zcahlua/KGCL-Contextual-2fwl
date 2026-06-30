import pytest

torch = pytest.importorskip("torch")

from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.chemistry.graphs import Vocab
from kgcl_retro.chemistry.sparse_pair_builder import SparsePairMetadata
from kgcl_retro.data.collate import ContextualEditTarget, GraphBatch
from kgcl_retro.models import KGCL


def _make_contextual_model():
    atom_vocab = Vocab([("Change Atom", (0, 0)), ("Attaching LG", (1,))])
    bond_vocab = Vocab([("Delete Bond", (None, None)), ("Change Bond", (1.0, None))])
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
    return KGCL(config=config, atom_vocab=atom_vocab, bond_vocab=bond_vocab)


def _make_graph_batch():
    pairs = torch.tensor([[1, 1], [1, 2], [2, 1], [2, 2]], dtype=torch.long)
    candidate_pairs = torch.tensor([[1, 2]], dtype=torch.long)
    sparse = SparsePairMetadata(
        enc_score_pairs=torch.zeros((0, 2), dtype=torch.long),
        dec_score_pairs_base=pairs,
        enc_carrier_pairs=torch.zeros((0, 2), dtype=torch.long),
        dec_carrier_pairs_base=pairs,
        enc_pair_scope=[(0, 0)],
        dec_pair_scope=[(0, int(pairs.size(0)))],
        enc_bridge_index=torch.zeros((0, 0), dtype=torch.long),
        enc_bridge_mask=torch.zeros((0, 0), dtype=torch.bool),
        dec_bridge_index_base=torch.zeros((0, 0), dtype=torch.long),
        dec_bridge_mask_base=torch.zeros((0, 0), dtype=torch.bool),
        pair_relation_codes=torch.zeros((0,), dtype=torch.long),
        dec_pair_relation_codes=torch.zeros((pairs.size(0),), dtype=torch.long),
        unordered_dec_candidate_pairs=candidate_pairs,
        action_pair_scope=[(0, 1)],
        atom_scope=[(1, 2)],
        dec_pair_relation_features=torch.zeros((pairs.size(0), 32)),
    )
    return GraphBatch(base_tensors=(), scopes=([], []), sparse_metadata=sparse, model_variant="contextual_2fwl")


def test_vectorized_action_scores_match_reference():
    model = _make_contextual_model().contextual_2fwl
    graph_batch = _make_graph_batch()
    torch.manual_seed(17)
    atom_states = torch.randn((3, 8))
    atom_fg_context = torch.randn((3, 8))
    pair_states = torch.randn((4, 8))
    pair_rel = torch.randn((4, 4))

    expected, expected_graph = model._score_actions_reference(atom_states, atom_fg_context, pair_states, pair_rel, graph_batch)
    actual, actual_graph = model._score_actions(atom_states, atom_fg_context, pair_states, pair_rel, graph_batch)

    assert torch.allclose(actual[0], expected[0], atol=1e-6)
    assert torch.allclose(actual_graph, expected_graph, atol=1e-6)


def test_action_order_unchanged():
    model = _make_contextual_model().contextual_2fwl
    graph_batch = _make_graph_batch()
    atom_states = torch.randn((3, 8))
    atom_fg_context = torch.randn((3, 8))
    pair_states = torch.randn((4, 8))
    pair_rel = torch.randn((4, 4))

    scores, _graph = model._score_actions(atom_states, atom_fg_context, pair_states, pair_rel, graph_batch)

    assert scores[0].numel() == model.bond_outdim + 2 * model.atom_outdim + 1
    assert graph_batch.sparse_metadata.action_vector_lengths == [scores[0].numel()]


def test_map_gold_target_indices_still_aligns():
    model = _make_contextual_model()
    graph_batch = _make_graph_batch()
    target = ContextualEditTarget(
        edit_type="bond",
        edit_class=("Change Bond", (1.0, None)),
        atom_maps=[1, 2],
        bond_class_index=1,
        gold_bond_pair=(1, 2),
    )

    indices = model.map_contextual_targets([target], graph_batch)

    assert indices.tolist() == [1]
