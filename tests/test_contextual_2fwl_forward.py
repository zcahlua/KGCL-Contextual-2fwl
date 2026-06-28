import pytest

Chem = pytest.importorskip("rdkit.Chem")
torch = pytest.importorskip("torch")

from kgcl_retro.chemistry.contextual_fg import FunctionalGroupInstance, MoleculeFGMetadata
from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.chemistry.graphs import MolGraph, Vocab
from kgcl_retro.data.collate import get_batch_graphs, prepare_contextual_edit_labels
from kgcl_retro.models import KGCL
from kgcl_retro.models.contextual_fg import ContextualFGOutput


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


def test_contextual_forward_builds_dynamic_targets_and_proposal_bce():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    graph = MolGraph(mol=mol, model_variant="contextual_2fwl", use_contextual_fg=True)
    graph_batch = get_batch_graphs([graph], model_variant="contextual_2fwl", pair_near_radius=1)
    atom_vocab = Vocab([("Change Atom", (0, 0))])
    bond_vocab = Vocab([("Add Bond", (1.0, None))])
    labels = prepare_contextual_edit_labels(
        [graph],
        [("Add Bond", (1.0, None))],
        [[1, 3]],
        bond_vocab,
        atom_vocab,
        graph_batch.sparse_metadata,
    )
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
        "pair_topk": 1,
    }
    model = KGCL(config=config, atom_vocab=atom_vocab, bond_vocab=bond_vocab)
    model.train()

    scores, _graph_vecs = model([graph_batch], [labels])
    target_indices = model.map_contextual_targets(labels, graph_batch)

    assert target_indices.shape == (1,)
    assert target_indices[0].item() < scores[0][0].numel()
    assert model.contextual_2fwl.last_proposal_loss is not None
    assert model.contextual_2fwl.last_proposal_loss.item() >= 0.0


def test_pair_use_proposal_false_keeps_decoder_on_encoder_pairs():
    mol = Chem.MolFromSmiles("[CH3:1][CH2:2][OH:3]")
    graph = MolGraph(mol=mol, model_variant="contextual_2fwl", use_contextual_fg=True)
    graph_batch = get_batch_graphs([graph], model_variant="contextual_2fwl", pair_near_radius=1)
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
        "pair_topk": 100,
        "pair_use_proposal": False,
    }
    model = KGCL(config=config, atom_vocab=atom_vocab, bond_vocab=bond_vocab)

    model([graph_batch])

    candidates = {tuple(pair) for pair in graph_batch.sparse_metadata.unordered_dec_candidate_pairs.tolist()}
    assert (1, 3) not in candidates
    assert model.contextual_2fwl.last_proposal_loss is not None
    assert model.contextual_2fwl.last_proposal_loss.item() == 0.0


def test_contextual_edge_gru_uses_previous_edge_state_as_hidden():
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

    class RecordingGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, input_tensor, hidden_tensor):
            self.calls.append((input_tensor.detach().clone(), hidden_tensor.detach().clone()))
            return hidden_tensor

    recorder = RecordingGRU()
    model.contextual_2fwl.edge_gru = recorder

    model.contextual_2fwl._encode_contextual_graph(graph_batch)

    edge_initial = model.contextual_2fwl.edge_input(graph_batch.base_tensors[1])
    assert torch.allclose(recorder.calls[0][1], edge_initial)


def test_pair_fg_context_pools_either_and_shared_fg_instances():
    atom_vocab = Vocab([("Change Atom", (0, 0))])
    bond_vocab = Vocab([("Add Bond", (1.0, None))])
    config = {
        "model_variant": "contextual_2fwl",
        "n_atom_feat": ATOM_FDIM,
        "n_bond_feat": ATOM_FDIM + BOND_FDIM,
        "mpn_size": 4,
        "mlp_size": 8,
        "depth": 1,
        "dropout_mlp": 0.0,
        "dropout_mpn": 0.0,
        "atom_message": False,
        "use_attn": False,
        "n_heads": 1,
        "fg_hidden_size": 4,
        "pair_hidden_size": 4,
        "pair_relation_size": 4,
        "pair_enc_layers": 1,
        "pair_dec_layers": 1,
        "pair_topk": 8,
    }
    model = KGCL(config=config, atom_vocab=atom_vocab, bond_vocab=bond_vocab).contextual_2fwl
    with torch.no_grad():
        model.fg_pair_pool_project.weight.copy_(torch.eye(4))
        model.fg_pair_pool_project.bias.zero_()
        model.null_pair_fg_either.copy_(torch.tensor([-1.0, -1.0, -1.0, -1.0]))
        model.null_pair_fg_shared.copy_(torch.tensor([-2.0, -2.0, -2.0, -2.0]))

    metadata = MoleculeFGMetadata(
        instances=[
            FunctionalGroupInstance("fg0", 0, (0,), (0,), {}, (), (), None, None),
            FunctionalGroupInstance("fg1", 1, (1,), (0, 1), {}, (), (), None, None),
            FunctionalGroupInstance("fg2", 2, (2,), (1,), {}, (), (), None, None),
        ],
        atom_to_fg_core=[[], [], []],
        atom_to_fg_context=[[0, 1], [1, 2], []],
        atom_fg_distance=[[], [], []],
        has_null=False,
        num_atoms=3,
    )
    fg_out = ContextualFGOutput(
        fg_embeddings=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
            ]
        ),
        atom_fg_context=torch.zeros((4, 4)),
        enhanced_atom_states=torch.zeros((4, 4)),
        fg_instance_scope=[(0, 3)],
        diagnostics={},
    )
    atom_fg_context = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0, 3.0],
            [5.0, 5.0, 5.0, 5.0],
        ]
    )
    pair_features = model._fg_pair_context(
        atom_fg_context,
        torch.tensor([[1, 2], [1, 3]], dtype=torch.long),
        fg_out=fg_out,
        fg_metadata=[metadata],
        atom_scope=[(1, 3)],
    )

    assert torch.allclose(pair_features[0, 12:16], torch.tensor([1.0, 2.0, 3.0, 0.0]))
    assert torch.allclose(pair_features[0, 16:20], torch.tensor([0.0, 2.0, 0.0, 0.0]))
    assert torch.allclose(pair_features[1, 12:16], torch.tensor([1.0, 2.0, 0.0, 0.0]))
    assert torch.allclose(pair_features[1, 16:20], torch.tensor([-2.0, -2.0, -2.0, -2.0]))
