import pytest

Chem = pytest.importorskip("rdkit.Chem")
torch = pytest.importorskip("torch")

from kgcl_retro.chemistry.contextual_fg import match_functional_group_instances
from kgcl_retro.chemistry.sparse_pair_builder import (
    PAIR_RELATION_FEATURE_SIZE,
    build_decoder_pair_metadata,
    build_encoder_pair_metadata,
    build_proposal_universe,
    build_sparse_pair_metadata,
)


def _pairs(tensor):
    return {tuple(row) for row in tensor.tolist()}


def test_pair_sets_include_diagonals_and_reversals():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    metadata = build_sparse_pair_metadata(
        mol,
        fg_metadata,
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
    )

    enc_pairs = _pairs(metadata.enc_score_pairs)
    assert {(1, 1), (2, 2), (3, 3)}.issubset(enc_pairs)
    for i, j in enc_pairs:
        assert (j, i) in enc_pairs


def test_score_pairs_are_subset_of_carrier_pairs():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    metadata = build_sparse_pair_metadata(mol, fg_metadata, atom_offset=1, build_decoder_base=True)

    assert _pairs(metadata.enc_score_pairs).issubset(_pairs(metadata.enc_carrier_pairs))
    assert _pairs(metadata.dec_score_pairs_base).issubset(_pairs(metadata.dec_carrier_pairs_base))


def test_bridge_indices_reference_closed_carrier_pairs():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    metadata = build_sparse_pair_metadata(
        mol,
        fg_metadata,
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=2,
        pair_max_bridges_enc=4,
    )

    carrier_pairs = _pairs(metadata.enc_carrier_pairs)
    for pair_idx, (i, j) in enumerate(metadata.enc_carrier_pairs.tolist()):
        for bridge_atom, enabled in zip(
            metadata.enc_bridge_index[pair_idx].tolist(),
            metadata.enc_bridge_mask[pair_idx].tolist(),
        ):
            if enabled:
                assert (i, bridge_atom) in carrier_pairs
                assert (bridge_atom, j) in carrier_pairs


def test_proposal_topk_expands_decoder_candidates():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)
    encoder = build_encoder_pair_metadata(
        mol,
        fg_metadata,
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
    )

    decoder = build_decoder_pair_metadata(
        mol,
        fg_metadata,
        enc_score_pairs=_pairs(encoder.enc_score_pairs),
        proposal_topk_pairs={(1, 3)},
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
        training=False,
    )

    assert (1, 3) in _pairs(decoder.dec_score_pairs_base)
    assert (3, 1) in _pairs(decoder.dec_score_pairs_base)
    assert (1, 3) in _pairs(decoder.unordered_dec_candidate_pairs)


def test_gold_pair_train_only_not_in_inference_candidates():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)
    encoder = build_encoder_pair_metadata(
        mol,
        fg_metadata,
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
    )

    inference = build_decoder_pair_metadata(
        mol,
        fg_metadata,
        enc_score_pairs=_pairs(encoder.enc_score_pairs),
        proposal_topk_pairs=set(),
        gold_bond_pairs={(1, 3)},
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
        training=False,
    )
    training = build_decoder_pair_metadata(
        mol,
        fg_metadata,
        enc_score_pairs=_pairs(encoder.enc_score_pairs),
        proposal_topk_pairs=set(),
        gold_bond_pairs={(1, 3)},
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
        training=True,
    )

    assert (1, 3) not in _pairs(inference.unordered_dec_candidate_pairs)
    assert (1, 3) in _pairs(training.unordered_dec_candidate_pairs)
    assert training.diagnostics["gold_pairs_rescued_by_teacher_forcing"] == 1


def test_training_gold_pair_survives_decoder_score_cap():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)
    encoder = build_encoder_pair_metadata(
        mol,
        fg_metadata,
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
    )

    training = build_decoder_pair_metadata(
        mol,
        fg_metadata,
        enc_score_pairs=_pairs(encoder.enc_score_pairs),
        proposal_topk_pairs=set(),
        gold_bond_pairs={(1, 3)},
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
        pair_max_score_pairs_dec=7,
        training=True,
    )

    assert (1, 3) in _pairs(training.unordered_dec_candidate_pairs)


def test_gold_only_pairs_do_not_bridge_unrelated_decoder_candidates():
    mol = Chem.MolFromSmiles("CCCC")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)
    encoder = build_encoder_pair_metadata(
        mol,
        fg_metadata,
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
    )

    decoder = build_decoder_pair_metadata(
        mol,
        fg_metadata,
        enc_score_pairs=_pairs(encoder.enc_score_pairs),
        proposal_topk_pairs={(1, 4)},
        gold_bond_pairs={(1, 3)},
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=2,
        pair_max_bridges_dec=4,
        training=True,
    )

    pair_lookup = {tuple(pair): idx for idx, pair in enumerate(decoder.dec_carrier_pairs_base.tolist())}
    pair_idx = pair_lookup[(1, 4)]
    bridges = {
        int(bridge)
        for bridge, enabled in zip(
            decoder.dec_bridge_index_base[pair_idx].tolist(),
            decoder.dec_bridge_mask_base[pair_idx].tolist(),
        )
        if enabled
    }
    assert 3 not in bridges


def test_proposal_bridges_added_to_decoder():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)
    encoder = build_encoder_pair_metadata(
        mol,
        fg_metadata,
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
    )

    decoder = build_decoder_pair_metadata(
        mol,
        fg_metadata,
        enc_score_pairs=_pairs(encoder.enc_score_pairs),
        proposal_topk_pairs={(1, 3), (1, 2), (2, 3)},
        atom_offset=1,
        pair_near_radius=1,
        pair_bridge_radius=1,
        pair_max_bridges_dec=4,
        training=False,
    )

    pair_lookup = {tuple(pair): idx for idx, pair in enumerate(decoder.dec_carrier_pairs_base.tolist())}
    pair_idx = pair_lookup[(1, 3)]
    bridges = {
        int(bridge)
        for bridge, enabled in zip(
            decoder.dec_bridge_index_base[pair_idx].tolist(),
            decoder.dec_bridge_mask_base[pair_idx].tolist(),
        )
        if enabled
    }
    assert 2 in bridges


def test_pair_relation_features_are_fixed_width_for_all_pairs():
    mol = Chem.MolFromSmiles("C1=CC=CC=C1O")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)
    proposal = build_proposal_universe(mol, fg_metadata, atom_offset=1, pair_near_radius=2)
    encoder = build_encoder_pair_metadata(mol, fg_metadata, atom_offset=1)

    assert encoder.pair_relation_features.shape[1] == PAIR_RELATION_FEATURE_SIZE
    assert proposal.pair_relation_features.shape[1] == PAIR_RELATION_FEATURE_SIZE
    assert encoder.pair_relation_features.size(0) == encoder.enc_carrier_pairs.size(0)
