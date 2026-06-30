import pytest

Chem = pytest.importorskip("rdkit.Chem")

from kgcl_retro.chemistry.contextual_fg import match_functional_group_instances
from kgcl_retro.chemistry.sparse_pair_builder import (
    PairBuilderCache,
    _distances,
    _relation_feature,
    build_pair_builder_cache,
)


def test_cache_distances_match_uncached():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    cache = build_pair_builder_cache(mol, fg_metadata, atom_offset=1)

    assert cache.distances == _distances(mol)


def test_cache_fg_context_membership():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    cache = build_pair_builder_cache(mol, fg_metadata, atom_offset=1)

    for atom_idx, expected in enumerate(fg_metadata.atom_to_fg_context):
        assert cache.atom_to_fg_context_abs[atom_idx] == frozenset(expected)
    assert all(isinstance(context, frozenset) for context in cache.fg_contexts_abs)


def test_cache_same_ring_matrix():
    mol = Chem.MolFromSmiles("C1CCCCC1")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    cache = build_pair_builder_cache(mol, fg_metadata, atom_offset=1)

    assert cache.same_ring_matrix[0][3] is True


def test_cache_same_aromatic_system_matrix():
    mol = Chem.MolFromSmiles("c1ccccc1O")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    cache = build_pair_builder_cache(mol, fg_metadata, atom_offset=1)

    assert cache.same_aromatic_system_matrix[0][3] is True


def test_cache_bond_feature_lookup():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    cache = build_pair_builder_cache(mol, fg_metadata, atom_offset=1)

    assert (1, 2) in cache.bond_exists
    assert (2, 1) in cache.bond_exists
    assert (1, 3) not in cache.bond_exists
    assert cache.bond_feature_map[(1, 2)]


def test_cache_relation_feature_matches_uncached_path_on_tiny_mols():
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)
    cache = build_pair_builder_cache(mol, fg_metadata, atom_offset=1)

    for pair in [(1, 1), (1, 2), (1, 3), (3, 1)]:
        cached = _relation_feature(mol, fg_metadata, cache.distances, pair[0], pair[1], 1, cache=cache)
        uncached = _relation_feature(mol, fg_metadata, cache.distances, pair[0], pair[1], 1)
        assert cached == uncached
