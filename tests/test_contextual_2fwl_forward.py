import pytest

Chem = pytest.importorskip("rdkit.Chem")
torch = pytest.importorskip("torch")

from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.chemistry.graphs import MolGraph, Vocab
from kgcl_retro.data.collate import get_batch_graphs
from kgcl_retro.models import KGCL


def test_contextual_2fwl_forward_tiny_molecule():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    graph = MolGraph(mol=mol, model_variant="contextual_2fwl", use_contextual_fg=True)
    graph_batch = get_batch_graphs([graph], model_variant="contextual_2fwl")
    atom_vocab = Vocab([("Change Atom", (0, 0))])
    bond_vocab = Vocab([("Delete Bond", (None, None)), ("Add Bond", (1.0, None))])
    config = {
        "model_variant": "contextual_2fwl",
        "n_atom_feat": ATOM_FDIM,
        "n_bond_feat": ATOM_FDIM + BOND_FDIM,
        "mpn_size": 32,
        "mlp_size": 64,
        "depth": 2,
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

    scores, graph_vecs = model([graph_batch])

    assert len(scores) == 1
    assert len(scores[0]) == 1
    assert scores[0][0].numel() == graph_batch.sparse_metadata.action_vector_lengths[0]
    assert graph_vecs[0].shape == (1, 32)
