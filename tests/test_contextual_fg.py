import pytest

Chem = pytest.importorskip("rdkit.Chem")
torch = pytest.importorskip("torch")

from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.chemistry.contextual_fg import match_functional_group_instances
from kgcl_retro.chemistry.graphs import MolGraph
from kgcl_retro.data.collate import get_batch_graphs
from kgcl_retro.models.contextual_fg import ContextualFGEncoder


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


def test_local_fg_encoder_uses_context_edges():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    graph = MolGraph(mol=mol, model_variant="contextual_2fwl", use_contextual_fg=True, fg_context_radius=1)
    graph_batch = get_batch_graphs([graph], model_variant="contextual_2fwl")
    encoder = ContextualFGEncoder(
        atom_hidden_size=8,
        fg_hidden_size=8,
        fg_layers=2,
        kg_embedding_size=8,
        bond_feature_size=ATOM_FDIM + BOND_FDIM,
    )
    atom_states = torch.randn(graph_batch.base_tensors[0].size(0), 8)
    atom_scope, _ = graph_batch.scopes

    base_out = encoder(atom_states, graph_batch.fg_metadata, atom_scope, graph_tensors=graph_batch.base_tensors)
    f_atoms, f_bonds, f_fgs, atom_num, n_mols, a2b, b2a, b2revb, undirected_b2a = graph_batch.base_tensors
    changed_bonds = f_bonds.clone()
    changed_bonds[1:] = changed_bonds[1:] + 1.0
    changed_tensors = (f_atoms, changed_bonds, f_fgs, atom_num, n_mols, a2b, b2a, b2revb, undirected_b2a)
    changed_out = encoder(atom_states, graph_batch.fg_metadata, atom_scope, graph_tensors=changed_tensors)

    assert not torch.allclose(base_out.fg_embeddings, changed_out.fg_embeddings)


def test_contextual_fg_pool_honors_mean_mode():
    encoder = ContextualFGEncoder(
        atom_hidden_size=2,
        fg_hidden_size=2,
        fg_layers=1,
        kg_embedding_size=2,
        fg_pool="mean",
    )
    atom_states = torch.tensor([[2.0, 4.0], [6.0, 8.0]])

    pooled = encoder._pool(atom_states, [0, 1], torch.zeros(2))

    assert torch.allclose(pooled, torch.tensor([4.0, 6.0]))
