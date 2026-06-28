import pytest

Chem = pytest.importorskip("rdkit.Chem")
torch = pytest.importorskip("torch")

from kgcl_retro.chemistry.graphs import MolGraph, Vocab
from kgcl_retro.data.collate import (
    get_batch_graphs,
    prepare_contextual_edit_labels,
)


def test_contextual_labels_store_gold_action_metadata():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    graph = MolGraph(
        mol=mol,
        model_variant="contextual_2fwl",
        use_contextual_fg=True,
    )
    graph_batch = get_batch_graphs([graph], model_variant="contextual_2fwl")
    bond_vocab = Vocab([("Add Bond", (1.0, None))])
    atom_vocab = Vocab([("Change Atom", (0, 0))])

    labels = prepare_contextual_edit_labels(
        [graph],
        [("Add Bond", (1.0, None))],
        [[1, 3]],
        bond_vocab,
        atom_vocab,
        graph_batch.sparse_metadata,
    )

    assert len(labels) == 1
    assert labels[0].edit_type == "bond"
    assert labels[0].edit_class == ("Add Bond", (1.0, None))
    assert labels[0].gold_bond_pair == (1, 3)
    assert labels[0].bond_class_index == 0


def test_contextual_labels_raise_when_gold_atom_missing():
    mol = Chem.MolFromSmiles("[CH3:1][OH:2]")
    graph = MolGraph(mol=mol, model_variant="contextual_2fwl", use_contextual_fg=True)
    graph_batch = get_batch_graphs([graph], model_variant="contextual_2fwl")
    bond_vocab = Vocab([("Add Bond", (1.0, None))])
    atom_vocab = Vocab([("Change Atom", (0, 0))])

    with pytest.raises(ValueError, match="Gold bond pair"):
        prepare_contextual_edit_labels(
            [graph],
            [("Add Bond", (1.0, None))],
            [[1, 99]],
            bond_vocab,
            atom_vocab,
            graph_batch.sparse_metadata,
        )
