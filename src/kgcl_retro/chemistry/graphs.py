from typing import List, Tuple  # Explanation: imports selected names needed to represent reaction graphs and inject functional-group knowledge
from rdkit import Chem  # Explanation: imports selected names needed to represent reaction graphs and inject functional-group knowledge
from kgcl_retro.chemistry.features import get_atom_features, get_bond_features  # Explanation: imports packaged atom and bond featurizers for graph construction.
from kgcl_retro.chemistry.functional_groups import load_functional_group_resources  # Explanation: imports the package-resource loader for KG functional-group embeddings.
import torch  # Explanation: imports torch for represent reaction graphs and inject functional-group knowledge
import torch.nn.functional as F  # Explanation: imports torch.nn.functional as F for represent reaction graphs and inject functional-group knowledge
import math  # Explanation: imports math for represent reaction graphs and inject functional-group knowledge


def match_fg(mol, use_rxn_class):  # Explanation: defines match_fg, which matches functional groups and retrieves KG embeddings
    embedding_set = "KGembedding_2" if use_rxn_class else "KGembedding"  # Explanation: selects the embedding table that matches the reaction-class setting.
    resources = load_functional_group_resources(embedding_set)  # Explanation: loads functional-group SMARTS and embeddings from packaged assets.
    fg_emb = []  # Explanation: collects KG embedding vectors for matched functional groups.
    fg_names = []  # Explanation: assigns an intermediate value used by later computation
    for sm in resources.smarts:  # Explanation: checks each functional-group SMARTS pattern against the molecule.
        if mol.HasSubstructMatch(sm):  # Explanation: handles the case where this molecule contains the functional group.
            name = resources.smarts_to_name[sm]  # Explanation: resolves the matched SMARTS pattern back to a functional-group name.
            fg_emb.append(resources.embeddings[name].tolist())  # Explanation: appends the KG embedding vector used for graph feature fusion.
            fg_names.append(name)  # Explanation: records the matched functional-group name for diagnostics.

    return fg_emb,fg_names  # Explanation: returns this computed result to the caller

def attention(query, key, mask=None, dropout=None):  # Explanation: defines attention, which fuses atom features with functional-group embeddings

    pad_rows = query.size(0) - key.size(0)  # Explanation: assigns an intermediate value used by later computation
    if pad_rows > 0:  # Explanation: checks this condition to choose the next execution path
        zero_padding = torch.zeros(pad_rows, key.size(1))  # Explanation: assigns an intermediate value used by later computation
        key_pad = key.clone()  # Explanation: assigns an intermediate value used by later computation
        key = torch.cat((key_pad, zero_padding), dim=0)  # Explanation: concatenates tensors along an existing dimension

    value = key.clone()  # Explanation: assigns an intermediate value used by later computation

    d_k = key.size(-1)  # Explanation: assigns an intermediate value used by later computation
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Explanation: assigns an intermediate value used by later computation

    p_attn = F.softmax(scores, dim=-1)  # Explanation: converts edit logits into probabilities
    if dropout is not None:  # Explanation: checks this condition to choose the next execution path
        p_attn = dropout(p_attn)  # Explanation: assigns an intermediate value used by later computation

    # Res coonect
    a = torch.matmul(p_attn, value)  # Explanation: assigns an intermediate value used by later computation
    out = query + torch.matmul(p_attn, value)  # Explanation: assigns an intermediate value used by later computation
    return out, p_attn  # Explanation: returns this computed result to the caller


class MolGraph:  # Explanation: defines MolGraph, single molecule graph with features and mappings
    """
    'MolGraph' represents the graph structure and featurization of a single molecule.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None, use_rxn_class: bool = False) -> None:  # Explanation: defines __init__, which represent reaction graphs and inject functional-group knowledge
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.mol = mol  # Explanation: stores this value on the object for later model operations
        self.rxn_class = rxn_class  # Explanation: stores this value on the object for later model operations
        self.use_rxn_class = use_rxn_class  # Explanation: stores this value on the object for later model operations
        self._build_mol()  # Explanation: uses or updates this object state during computation
        self._build_graph()  # Explanation: uses or updates this object state during computation

    def _build_mol(self) -> None:  # Explanation: defines _build_mol, which represent reaction graphs and inject functional-group knowledge
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()  # Explanation: stores this value on the object for later model operations
        self.num_bonds = self.mol.GetNumBonds()  # Explanation: stores this value on the object for later model operations
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()  # Explanation: stores this value on the object for later model operations
                            for atom in self.mol.GetAtoms()}  # Explanation: iterates over this collection to process each item
        self.idx_to_amap = {value: key for key,  # Explanation: stores this value on the object for later model operations
                                           value in self.amap_to_idx.items()}  # Explanation: executes this statement as part of represent reaction graphs and inject functional-group knowledge

    def _build_graph(self):  # Explanation: defines _build_graph, which represent reaction graphs and inject functional-group knowledge
        """Builds the graph attributes."""
        self.n_atoms = 0  # number of atoms  # Explanation: stores this value on the object for later model operations
        self.n_bonds = 0  # number of bonds  # Explanation: stores this value on the object for later model operations
        self.f_atoms = []  # mapping from atom index to atom features  # Explanation: stores this value on the object for later model operations
        # mapping from bond index to concat(in_atom, bond) features
        self.f_bonds = []  # Explanation: stores this value on the object for later model operations
        self.a2b = []  # mapping from atom index to incoming bond indices  # Explanation: stores this value on the object for later model operations
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from  # Explanation: stores this value on the object for later model operations
        self.b2revb = []  # mapping from bond index to the index of the reverse bond  # Explanation: stores this value on the object for later model operations

        # functional group embedding
        self.f_fgs, self.fg_names = match_fg(self.mol, self.use_rxn_class)  # Explanation: stores this value on the object for later model operations
        self.atoms = []  # Explanation: stores this value on the object for later model operations

        # Get atom features
        self.f_atoms = [get_atom_features(  # Explanation: stores this value on the object for later model operations
            atom, rxn_class=self.rxn_class, use_rxn_class=self.use_rxn_class) for atom in self.mol.GetAtoms()]  # Explanation: assigns an intermediate value used by later computation
        self.n_atoms = len(self.f_atoms)  # Explanation: stores this value on the object for later model operations
        for atom in self.mol.GetAtoms():  # Explanation: iterates over this collection to process each item
            self.atoms.append(atom.GetSymbol())  # Explanation: uses or updates this object state during computation
        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):  # Explanation: iterates over this collection to process each item
            self.a2b.append([])  # Explanation: uses or updates this object state during computation

        # add group knowledge
        if self.f_fgs:  # Explanation: checks this condition to choose the next execution path
            temp_tensor = torch.tensor(self.f_atoms)  # Explanation: assigns an intermediate value used by later computation
            f_fgs_tensor = torch.tensor(self.f_fgs)  # Explanation: assigns an intermediate value used by later computation
            fuse_f_atoms, self.attn_score = attention(temp_tensor, f_fgs_tensor)  # Explanation: assigns an intermediate value used by later computation
            self.f_atoms = fuse_f_atoms.tolist()  # Explanation: stores this value on the object for later model operations

        # Get bond features
        for a1 in range(self.n_atoms):  # Explanation: iterates over this collection to process each item
            for a2 in range(a1 + 1, self.n_atoms):  # Explanation: iterates over this collection to process each item
                bond = self.mol.GetBondBetweenAtoms(a1, a2)  # Explanation: assigns an intermediate value used by later computation

                if bond is None:  # Explanation: checks this condition to choose the next execution path
                    continue  # Explanation: skips the rest of this loop iteration

                f_bond = get_bond_features(bond)  # Explanation: assigns an intermediate value used by later computation

                self.f_bonds.append(self.f_atoms[a1] + f_bond)  # Explanation: uses or updates this object state during computation
                self.f_bonds.append(self.f_atoms[a2] + f_bond)  # Explanation: uses or updates this object state during computation

                # Update index mappings
                b1 = self.n_bonds  # Explanation: assigns an intermediate value used by later computation
                b2 = b1 + 1  # Explanation: assigns an intermediate value used by later computation
                self.a2b[a2].append(b1)  # b1 = a1 --> a2  # Explanation: stores this value on the object for later model operations
                self.b2a.append(a1)  # Explanation: uses or updates this object state during computation
                self.a2b[a1].append(b2)  # b2 = a2 --> a1  # Explanation: stores this value on the object for later model operations
                self.b2a.append(a2)  # Explanation: uses or updates this object state during computation
                self.b2revb.append(b2)  # Explanation: uses or updates this object state during computation
                self.b2revb.append(b1)  # Explanation: uses or updates this object state during computation
                self.n_bonds += 2  # Explanation: stores this value on the object for later model operations

class RxnGraph:  # Explanation: defines RxnGraph, reaction state containing a product graph and edit label
    """
    RxnGraph contains the information of a reaction, like reactants, products. The edits associated with the reaction are also captured in edit labels.
    """

    def __init__(self, prod_mol: Chem.Mol, edit_to_apply: Tuple, edit_atom: List = [], reac_mol: Chem.Mol = None,  # Explanation: defines __init__, which represent reaction graphs and inject functional-group knowledge
                 rxn_class: int = None, use_rxn_class: bool = False) -> None:  # Explanation: computes an intermediate value for molecular graph editing
        """
        Parameters
        ----------
        prod_mol: Chem.Mol,
            Product molecule
        reac_mol: Chem.Mol, default None
            Reactant molecule(s)
        edit_to_apply: Tuple,
            Edits to apply to the product molecule
        edit_atom: List,
            Edit atom of product molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional input
        """
        self.prod_graph = MolGraph(  # Explanation: stores this value on the object for later model operations
            mol=prod_mol, rxn_class=rxn_class, use_rxn_class=use_rxn_class)  # Explanation: assigns an intermediate value used by later computation
        if reac_mol is not None:  # Explanation: checks this condition to choose the next execution path
            self.reac_mol = reac_mol  # Explanation: stores this value on the object for later model operations
        self.edit_to_apply = edit_to_apply  # Explanation: stores this value on the object for later model operations
        self.edit_atom = edit_atom  # Explanation: stores this value on the object for later model operations
        self.rxn_class = rxn_class  # Explanation: stores this value on the object for later model operations

    def get_components(self, attrs: List = ['prod_graph', 'edit_to_apply', 'edit_atom']) -> Tuple:  # Explanation: defines get_components, which represent reaction graphs and inject functional-group knowledge
        """ 
        Returns the components associated with the reaction graph. 
        """
        attr_tuple = ()  # Explanation: assigns an intermediate value used by later computation
        for attr in attrs:  # Explanation: iterates over this collection to process each item
            if hasattr(self, attr):  # Explanation: checks this condition to choose the next execution path
                attr_tuple += (getattr(self, attr),)  # Explanation: assigns an intermediate value used by later computation
            else:  # Explanation: handles the fallback branch for the preceding condition
                print(f"Does not have attr {attr}")  # Explanation: prints progress or diagnostic information

        return attr_tuple  # Explanation: returns this computed result to the caller


class Vocab:  # Explanation: defines Vocab, maps edit tuples to integer ids
    """
    Vocab class to deal with vocabularies and other attributes.
    """

    def __init__(self, elem_list: List) -> None:  # Explanation: defines __init__, which represent reaction graphs and inject functional-group knowledge
        """
        Parameters
        ----------
        elem_list: List, default ATOM_LIST
            Element list used for setting up the vocab
        """
        self.elem_list = elem_list  # Explanation: stores this value on the object for later model operations
        if isinstance(elem_list, dict):  # Explanation: checks this condition to choose the next execution path
            self.elem_list = list(elem_list.keys())  # Explanation: stores this value on the object for later model operations
        self.elem_to_idx = {a: idx for idx, a in enumerate(self.elem_list)}  # Explanation: stores this value on the object for later model operations
        self.idx_to_elem = {idx: a for idx, a in enumerate(self.elem_list)}  # Explanation: stores this value on the object for later model operations

    def __getitem__(self, a_type: Tuple) -> int:  # Explanation: defines __getitem__, which represent reaction graphs and inject functional-group knowledge
        return self.elem_to_idx[a_type]  # Explanation: returns this computed result to the caller

    def get(self, elem: Tuple, idx: int = None) -> int:  # Explanation: defines get, which represent reaction graphs and inject functional-group knowledge
        """Returns the index of the element, else a None for missing element.

        Parameters
        ----------
        elem: str,
            Element to query
        idx: int, default None
            Index to return if element not in vocab
        """
        return self.elem_to_idx.get(elem, idx)  # Explanation: returns this computed result to the caller

    def get_elem(self, idx: int) -> Tuple:  # Explanation: defines get_elem, which represent reaction graphs and inject functional-group knowledge
        """Returns the element at given index.

        Parameters
        ----------
        idx: int,
            Index to return if element not in vocab
        """
        return self.idx_to_elem[idx]  # Explanation: returns this computed result to the caller

    def __len__(self) -> int:  # Explanation: defines __len__, which represent reaction graphs and inject functional-group knowledge
        return len(self.elem_list)  # Explanation: returns this computed result to the caller

    def get_index(self, elem: Tuple) -> int:  # Explanation: defines get_index, which represent reaction graphs and inject functional-group knowledge
        """Returns the index of the element.

        Parameters
        ----------
        elem: str,
            Element to query
        """
        return self.elem_to_idx[elem]  # Explanation: returns this computed result to the caller

    def size(self) -> int:  # Explanation: defines size, which represent reaction graphs and inject functional-group knowledge
        """Returns length of Vocab."""
        return len(self.elem_list)  # Explanation: returns this computed result to the caller
