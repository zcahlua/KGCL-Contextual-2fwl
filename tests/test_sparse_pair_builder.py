import pytest

Chem = pytest.importorskip("rdkit.Chem")
torch = pytest.importorskip("torch")

from kgcl_retro.chemistry.contextual_fg import match_functional_group_instances
from kgcl_retro.chemistry.sparse_pair_builder import build_sparse_pair_metadata


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

    metadata = build_sparse_pair_metadata(mol, fg_metadata, atom_offset=1)

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
