import pytest

Chem = pytest.importorskip("rdkit.Chem")
torch = pytest.importorskip("torch")

from kgcl_retro.chemistry.contextual_fg import match_functional_group_instances
from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.chemistry.graphs import MolGraph, Vocab
from kgcl_retro.chemistry.sparse_pair_builder import build_sparse_pair_metadata
from kgcl_retro.data.collate import get_batch_graphs
from kgcl_retro.models import KGCL


def test_build_sparse_pair_metadata_static_only_has_encoder_and_proposal():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    metadata = build_sparse_pair_metadata(mol, fg_metadata, atom_offset=1)

    assert metadata.enc_carrier_pairs.numel() > 0
    assert metadata.proposal_universe_pairs.numel() > 0


def test_static_only_has_empty_decoder_defaults():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    metadata = build_sparse_pair_metadata(mol, fg_metadata, atom_offset=1)

    assert metadata.dec_score_pairs_base.shape == (0, 2)
    assert metadata.dec_carrier_pairs_base.shape == (0, 2)
    assert metadata.unordered_dec_candidate_pairs.shape == (0, 2)
    assert metadata.action_pair_scope == [(0, 0)]


def test_build_decoder_base_true_preserves_old_debug_behavior():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    metadata = build_sparse_pair_metadata(mol, fg_metadata, atom_offset=1, build_decoder_base=True)

    assert metadata.dec_score_pairs_base.numel() > 0
    assert metadata.dec_carrier_pairs_base.numel() > 0
    assert metadata.unordered_dec_candidate_pairs.numel() > 0


def test_get_batch_graphs_defaults_to_static_only_prepared_metadata():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    graph = MolGraph(mol=mol, model_variant="contextual_2fwl", use_contextual_fg=True)

    graph_batch = get_batch_graphs([graph], model_variant="contextual_2fwl")

    assert graph_batch.sparse_metadata.dec_score_pairs_base.shape == (0, 2)
    assert graph_batch.sparse_metadata.proposal_universe_pairs.numel() > 0


def test_forward_works_with_static_only_prepared_metadata():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    graph = MolGraph(mol=mol, model_variant="contextual_2fwl", use_contextual_fg=True)
    graph_batch = get_batch_graphs([graph], model_variant="contextual_2fwl")
    atom_vocab = Vocab([("Change Atom", (0, 0))])
    bond_vocab = Vocab([("Add Bond", (1.0, None))])
    config = {
        "model_variant": "contextual_2fwl",
        "n_atom_feat": ATOM_FDIM,
        "n_bond_feat": ATOM_FDIM + BOND_FDIM,
        "mpn_size": 32,
        "mlp_size": 64,
        "depth": 1,
        "dropout_mlp": 0.0,
        "dropout_mpn": 0.0,
        "atom_message": False,
        "use_attn": False,
        "n_heads": 1,
        "fg_hidden_size": 32,
        "pair_hidden_size": 32,
        "pair_relation_size": 16,
        "pair_enc_layers": 1,
        "pair_dec_layers": 1,
        "pair_topk": 8,
    }
    model = KGCL(config=config, atom_vocab=atom_vocab, bond_vocab=bond_vocab)

    scores, _graph_vecs = model([graph_batch])

    assert scores[0][0].numel() == graph_batch.sparse_metadata.action_vector_lengths[0]
    assert graph_batch.sparse_metadata.dec_carrier_pairs_base.numel() > 0
