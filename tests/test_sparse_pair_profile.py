import importlib

import pytest

Chem = pytest.importorskip("rdkit.Chem")

from kgcl_retro.chemistry.contextual_fg import match_functional_group_instances


def test_profile_disabled_by_default(monkeypatch):
    monkeypatch.delenv("KGCL_PROFILE_SPARSE_PAIR", raising=False)
    builder = importlib.reload(importlib.import_module("kgcl_retro.chemistry.sparse_pair_builder"))
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    builder.reset_profile_stats()
    builder.build_sparse_pair_metadata(mol, fg_metadata)

    assert builder.get_profile_stats() == {}


def test_profile_counts_distances_once_per_molecule(monkeypatch):
    monkeypatch.setenv("KGCL_PROFILE_SPARSE_PAIR", "1")
    builder = importlib.reload(importlib.import_module("kgcl_retro.chemistry.sparse_pair_builder"))
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    builder.reset_profile_stats()
    builder.build_sparse_pair_metadata(mol, fg_metadata)
    stats = builder.get_profile_stats()

    assert stats["_distances"]["calls"] == 1
    assert stats["build_sparse_pair_metadata"]["calls"] == 1


def test_profile_records_bridge_and_relation_calls(monkeypatch):
    monkeypatch.setenv("KGCL_PROFILE_SPARSE_PAIR", "1")
    builder = importlib.reload(importlib.import_module("kgcl_retro.chemistry.sparse_pair_builder"))
    mol = Chem.MolFromSmiles("CCO")
    fg_metadata = match_functional_group_instances(mol, False, radius=1)

    builder.reset_profile_stats()
    builder.build_sparse_pair_metadata(mol, fg_metadata, build_decoder_base=True)
    stats = builder.get_profile_stats()

    assert stats["_bridge_atoms"]["calls"] > 0
    assert stats["_relation_feature"]["calls"] > 0
    assert stats["_relation_tensors"]["calls"] > 0
