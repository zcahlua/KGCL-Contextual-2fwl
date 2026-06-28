from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, List, Tuple  # Explanation: imports selected names needed to collate molecular graphs and edit labels into tensors
import numpy as np  # Explanation: imports numpy as np for collate molecular graphs and edit labels into tensors
import torch  # Explanation: imports torch for collate molecular graphs and edit labels into tensors
from rdkit import Chem
from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM  # Explanation: imports packaged atom and bond feature dimensions for label tensor sizing.
from kgcl_retro.chemistry.graphs import MolGraph  # Explanation: imports the packaged molecule graph class used by collate functions.
from kgcl_retro.chemistry.contextual_fg import MoleculeFGMetadata
from kgcl_retro.chemistry.sparse_pair_builder import (
    SparsePairMetadata,
    build_sparse_pair_metadata,
    merge_sparse_pair_metadata,
)
from kgcl_retro.config.schema import normalize_model_variant


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return replace(obj, **{field.name: _move_to_device(getattr(obj, field.name), device) for field in fields(obj)})
    if isinstance(obj, tuple):
        return tuple(_move_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [_move_to_device(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    return obj


@dataclass
class GraphBatch:
    base_tensors: tuple[torch.Tensor, ...]
    scopes: tuple[list[tuple[int, int]], list[tuple[int, int]]]
    fg_metadata: list[MoleculeFGMetadata] | None = None
    sparse_metadata: SparsePairMetadata | None = None
    model_variant: str = "kgcl"
    mols: list[Any] | None = None

    def to(self, device):
        return GraphBatch(
            base_tensors=_move_to_device(self.base_tensors, device),
            scopes=self.scopes,
            fg_metadata=self.fg_metadata,
            sparse_metadata=_move_to_device(self.sparse_metadata, device),
            model_variant=self.model_variant,
            mols=self.mols,
        )


@dataclass(frozen=True)
class ContextualEditTarget:
    edit_type: str
    edit_class: Any
    atom_maps: Any
    bond_class_index: int | None = None
    atom_class_index: int | None = None
    gold_bond_pair: tuple[int, int] | None = None
    gold_atom_index: int | None = None
    stop: bool = False

def create_pad_tensor(alist):  # Explanation: defines create_pad_tensor, which pads variable-length index lists
    max_len = max([len(a) for a in alist])  # Explanation: assigns an intermediate value used by later computation
    for a in alist:  # Explanation: iterates over this collection to process each item
        pad_len = max_len - len(a)  # Explanation: assigns an intermediate value used by later computation
        a.extend([0] * pad_len)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
    return torch.tensor(alist, dtype=torch.long)  # Explanation: returns this computed result to the caller


def prepare_edit_labels(graph_batch: List[MolGraph], edits: List[Any], edit_atoms: List[Any], bond_vocab: List, atom_vocab: List) -> torch.tensor:  # Explanation: defines prepare_edit_labels, which creates flattened one-hot edit labels
    """ 
    Prepare edit label including atom edits and bond edits.
    """
    bond_vocab_size = bond_vocab.size()  # Explanation: computes an intermediate value for molecular graph editing
    atom_vocab_size = atom_vocab.size()  # Explanation: computes an intermediate value for molecular graph editing
    edit_labels = []  # Explanation: computes an intermediate value for molecular graph editing

    for prod_graph, edit, edit_atom in zip(graph_batch, edits, edit_atoms):  # Explanation: iterates over this collection to process each item
        bond_label = np.zeros((prod_graph.num_bonds, bond_vocab_size))  # Explanation: computes an intermediate value for molecular graph editing
        atom_label = np.zeros((prod_graph.num_atoms, atom_vocab_size))  # Explanation: computes an intermediate value for molecular graph editing
        stop_label = np.zeros((1,))  # Explanation: assigns an intermediate value used by later computation

        if edit == 'Terminate':  # Explanation: checks this condition to choose the next execution path
            stop_label[0] = 1.0  # Explanation: assigns an intermediate value used by later computation

        elif edit[0] == 'Change Atom' or edit[0] == 'Attaching LG':  # Explanation: checks an alternate condition after the previous branch failed
            a_map = edit_atom  # Explanation: assigns an intermediate value used by later computation
            a_idx = prod_graph.amap_to_idx[a_map]  # Explanation: assigns an intermediate value used by later computation
            edit_idx = atom_vocab.get_index(edit)  # Explanation: computes an intermediate value for molecular graph editing
            atom_label[a_idx][edit_idx] = 1  # Explanation: computes an intermediate value for molecular graph editing

        else:  # Explanation: handles the fallback branch for the preceding condition
            a1, a2 = edit_atom[0], edit_atom[1]  # Explanation: assigns an intermediate value used by later computation
            a_start, a_end = prod_graph.amap_to_idx[a1], prod_graph.amap_to_idx[a2]  # Explanation: assigns an intermediate value used by later computation
            b_idx = prod_graph.mol.GetBondBetweenAtoms(a_start, a_end).GetIdx()  # Explanation: assigns an intermediate value used by later computation
            edit_idx = bond_vocab.get_index(edit)  # Explanation: computes an intermediate value for molecular graph editing
            bond_label[b_idx][edit_idx] = 1  # Explanation: computes an intermediate value for molecular graph editing

        edit_label = np.concatenate(  # Explanation: computes an intermediate value for molecular graph editing
            (bond_label.flatten(), atom_label.flatten(), stop_label.flatten()))  # Explanation: continues a structured literal or expression
        edit_label = torch.from_numpy(edit_label)  # Explanation: computes an intermediate value for molecular graph editing
        edit_labels.append(edit_label)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

    return edit_labels  # Explanation: returns this computed result to the caller


def prepare_contextual_edit_labels(
    graph_batch: List[MolGraph],
    edits: List[Any],
    edit_atoms: List[Any],
    bond_vocab: List,
    atom_vocab: List,
    sparse_metadata: SparsePairMetadata | None = None,
) -> list[ContextualEditTarget]:
    labels: list[ContextualEditTarget] = []
    gold_bond_pairs: list[tuple[int, int]] = []
    gold_atom_indices: list[int] = []

    for mol_idx, (prod_graph, edit, edit_atom) in enumerate(zip(graph_batch, edits, edit_atoms)):
        atom_start = sparse_metadata.atom_scope[mol_idx][0] if sparse_metadata is not None else 0

        if edit == "Terminate":
            label = ContextualEditTarget(edit_type="stop", edit_class=edit, atom_maps=edit_atom, stop=True)
        elif edit[0] == "Change Atom" or edit[0] == "Attaching LG":
            atom_map = edit_atom
            if atom_map not in prod_graph.amap_to_idx:
                raise ValueError(f"Gold atom map {atom_map} is not present in product graph.")
            local_atom_idx = prod_graph.amap_to_idx[atom_map]
            edit_idx = atom_vocab.get_index(edit)
            gold_atom_idx = atom_start + local_atom_idx
            label = ContextualEditTarget(
                edit_type="atom",
                edit_class=edit,
                atom_maps=edit_atom,
                atom_class_index=edit_idx,
                gold_atom_index=gold_atom_idx,
            )
            gold_atom_indices.append(gold_atom_idx)
        else:
            atom_map_1, atom_map_2 = edit_atom[0], edit_atom[1]
            if atom_map_1 not in prod_graph.amap_to_idx or atom_map_2 not in prod_graph.amap_to_idx:
                raise ValueError(f"Gold bond pair {edit_atom} contains atom maps absent from product graph.")
            pair = tuple(
                sorted(
                    [
                        atom_start + prod_graph.amap_to_idx[atom_map_1],
                        atom_start + prod_graph.amap_to_idx[atom_map_2],
                    ]
                )
            )
            edit_idx = bond_vocab.get_index(edit)
            label = ContextualEditTarget(
                edit_type="bond",
                edit_class=edit,
                atom_maps=edit_atom,
                bond_class_index=edit_idx,
                gold_bond_pair=pair,
            )
            gold_bond_pairs.append(pair)

        labels.append(label)

    if sparse_metadata is not None:
        sparse_metadata.gold_bond_pairs = (
            torch.tensor(gold_bond_pairs, dtype=torch.long)
            if gold_bond_pairs
            else torch.zeros((0, 2), dtype=torch.long)
        )
        sparse_metadata.gold_atom_indices = (
            torch.tensor(gold_atom_indices, dtype=torch.long)
            if gold_atom_indices
            else torch.zeros((0,), dtype=torch.long)
        )
    return labels


def get_batch_graphs(
    graph_batch: List[MolGraph],
    use_rxn_class: bool = False,
    model_variant: str = "kgcl",
    fg_context_radius: int = 1,
    pair_near_radius: int = 2,
    pair_bridge_radius: int = 2,
    pair_max_score_pairs_enc: int = 512,
    pair_max_score_pairs_dec: int = 1024,
    pair_max_carrier_pairs_enc: int = 1024,
    pair_max_carrier_pairs_dec: int = 2048,
    pair_max_bridges_enc: int = 8,
    pair_max_bridges_dec: int = 8,
    pair_topk: int = 64,
) -> Tuple[torch.Tensor, List[Tuple[int]]]:  # Explanation: defines get_batch_graphs, which builds batched molecular graph tensors
    """
    Featurization of a batch of molecules.
    """
    # Start n_atoms and n_bonds at 1 b/c zero padding
    n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)  # Explanation: assigns an intermediate value used by later computation
    n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)  # Explanation: assigns an intermediate value used by later computation
    a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule  # Explanation: assigns an intermediate value used by later computation
    b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule  # Explanation: assigns an intermediate value used by later computation

    # All start with zero padding so that indexing with zero padding returns zeros
    if use_rxn_class:  # Explanation: checks this condition to choose the next execution path
        atom_fdim = ATOM_FDIM + 10  # Explanation: computes an intermediate value for molecular graph editing
    else:  # Explanation: handles the fallback branch for the preceding condition
        atom_fdim = ATOM_FDIM  # Explanation: computes an intermediate value for molecular graph editing
    bond_fdim = atom_fdim + BOND_FDIM  # Explanation: computes an intermediate value for molecular graph editing

    f_atoms = [[0] * atom_fdim]  # atom features  # Explanation: assigns an intermediate value used by later computation
    f_bonds = [[0] * bond_fdim]  # combined atom/bond features  # Explanation: assigns an intermediate value used by later computation
    a2b = [[]]  # mapping from atom index to incoming bond indices  # Explanation: assigns an intermediate value used by later computation
    b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from  # Explanation: assigns an intermediate value used by later computation
    b2revb = [0]  # mapping from bond index to the index of the reverse bond  # Explanation: assigns an intermediate value used by later computation
    undirected_b2a = [[]]  # mapping from the undirected bond index to the beginindex and endindex of the atoms  # Explanation: assigns an intermediate value used by later computation
    n_mols = 0  # Explanation: assigns an intermediate value used by later computation
    f_fgs = []  # Explanation: assigns an intermediate value used by later computation
    atom_num = []  # Explanation: computes an intermediate value for molecular graph editing

    variant = normalize_model_variant(model_variant)
    fg_metadata = []
    sparse_items = []
    mols = []

    for mol_graph in graph_batch:  # Explanation: iterates over this collection to process each item

        f_atoms.extend(mol_graph.f_atoms)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        f_bonds.extend(mol_graph.f_bonds)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        n_mols += 1  # Explanation: assigns an intermediate value used by later computation

        for a in range(mol_graph.n_atoms):  # Explanation: iterates over this collection to process each item
            a2b.append([b + n_bonds for b in mol_graph.a2b[a]])  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

        for b in range(mol_graph.n_bonds):  # Explanation: iterates over this collection to process each item
            b2a.append(n_atoms + mol_graph.b2a[b])  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
            b2revb.append(n_bonds + mol_graph.b2revb[b])  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

        n_undirected_bonds = len(undirected_b2a)  # Explanation: assigns an intermediate value used by later computation
        for bond in mol_graph.mol.GetBonds():  # Explanation: iterates over this collection to process each item
            undirected_b2a.append(sorted([bond.GetBeginAtomIdx() + n_atoms, bond.GetEndAtomIdx() + n_atoms]))  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

        a_scope.append((n_atoms, mol_graph.n_atoms))  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        b_scope.append((n_undirected_bonds, mol_graph.num_bonds))  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        if variant == "contextual_2fwl":
            if not hasattr(mol_graph, "fg_metadata"):
                raise ValueError(
                    "MolGraph lacks contextual FG metadata. Build MolGraph with "
                    "model_variant='contextual_2fwl' before contextual batching."
                )
            fg_metadata.append(mol_graph.fg_metadata)
            mols.append(Chem.Mol(mol_graph.mol))
            sparse_items.append(
                build_sparse_pair_metadata(
                    mol_graph.mol,
                    mol_graph.fg_metadata,
                    atom_offset=n_atoms,
                    pair_near_radius=pair_near_radius,
                    pair_bridge_radius=pair_bridge_radius,
                    pair_max_score_pairs_enc=pair_max_score_pairs_enc,
                    pair_max_score_pairs_dec=pair_max_score_pairs_dec,
                    pair_max_carrier_pairs_enc=pair_max_carrier_pairs_enc,
                    pair_max_carrier_pairs_dec=pair_max_carrier_pairs_dec,
                    pair_max_bridges_enc=pair_max_bridges_enc,
                    pair_max_bridges_dec=pair_max_bridges_dec,
                    pair_topk=pair_topk,
                )
            )
        n_atoms += mol_graph.n_atoms  # Explanation: assigns an intermediate value used by later computation
        n_bonds += mol_graph.n_bonds  # Explanation: assigns an intermediate value used by later computation

        f_fgs.extend(mol_graph.f_fgs)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors
        atom_num.append(mol_graph.n_atoms)  # Explanation: executes this statement as part of collate molecular graphs and edit labels into tensors

    f_atoms = torch.FloatTensor(f_atoms)  # Explanation: assigns an intermediate value used by later computation
    f_bonds = torch.FloatTensor(f_bonds)  # Explanation: assigns an intermediate value used by later computation
    a2b = create_pad_tensor(a2b)  # Explanation: assigns an intermediate value used by later computation
    b2a = torch.LongTensor(b2a)  # Explanation: assigns an intermediate value used by later computation
    b2revb = torch.LongTensor(b2revb)  # Explanation: assigns an intermediate value used by later computation
    undirected_b2a = create_pad_tensor(undirected_b2a)  # Explanation: assigns an intermediate value used by later computation
    f_fgs = torch.FloatTensor(f_fgs)  # Explanation: assigns an intermediate value used by later computation
    atom_num = torch.tensor(atom_num)  # Explanation: computes an intermediate value for molecular graph editing
    n_mols = torch.tensor(n_mols)  # Explanation: assigns an intermediate value used by later computation

    graph_tensors = (f_atoms, f_bonds, f_fgs, atom_num, n_mols, a2b, b2a, b2revb, undirected_b2a)  # Explanation: computes an intermediate value for molecular graph editing
    scopes = (a_scope, b_scope)  # Explanation: assigns an intermediate value used by later computation

    if variant == "contextual_2fwl":
        return GraphBatch(
            base_tensors=graph_tensors,
            scopes=scopes,
            fg_metadata=fg_metadata,
            sparse_metadata=merge_sparse_pair_metadata(sparse_items),
            model_variant=variant,
            mols=mols,
        )

    return graph_tensors, scopes  # Explanation: returns this computed result to the caller
    
