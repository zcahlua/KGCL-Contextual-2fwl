import pytest

Chem = pytest.importorskip("rdkit.Chem")
pytest.importorskip("torch")

from kgcl_retro.chemistry.contextual_fg import match_functional_group_instances
from kgcl_retro.chemistry.graphs import MolGraph


def test_contextual_fg_null_token_for_molecule_without_matches():
    mol = Chem.MolFromSmiles("[He]")

    metadata = match_functional_group_instances(
        mol,
        use_rxn_class=False,
        radius=1,
        include_null=True,
    )

    assert metadata.has_null is True
    assert len(metadata.instances) == 1
    assert metadata.instances[0].is_null is True
    assert metadata.atom_to_fg_core == [[]]


def test_contextual_fg_allows_overlapping_instances():
    mol = Chem.MolFromSmiles("c1ccccc1O")

    metadata = match_functional_group_instances(
        mol,
        use_rxn_class=False,
        radius=1,
        include_null=False,
    )

    atom_memberships = [len(groups) for groups in metadata.atom_to_fg_core]
    assert max(atom_memberships) > 1


def test_context_radius_includes_neighbor_atoms():
    mol = Chem.MolFromSmiles("CCO")

    metadata = match_functional_group_instances(
        mol,
        use_rxn_class=False,
        radius=1,
        include_null=False,
    )

    hydroxyl = next(inst for inst in metadata.instances if inst.fg_name == "Hydroxyl")
    assert set(hydroxyl.core_atom_indices).issubset(hydroxyl.context_atom_indices)
    assert len(hydroxyl.context_atom_indices) > len(hydroxyl.core_atom_indices)


def test_molgraph_contextual_mode_skips_inline_fg_attention():
    mol = Chem.MolFromSmiles("CCO")

    baseline = MolGraph(mol=Chem.Mol(mol), use_rxn_class=False)
    contextual = MolGraph(
        mol=Chem.Mol(mol),
        use_rxn_class=False,
        model_variant="contextual_2fwl",
        use_contextual_fg=True,
        fg_context_radius=1,
    )

    assert hasattr(contextual, "fg_metadata")
    assert contextual.f_atoms != baseline.f_atoms
    assert not hasattr(contextual, "attn_score")
