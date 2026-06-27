from typing import Dict, List, Tuple, Union  # Explanation: imports selected names needed to define the KGCL graph-edit prediction model

import torch  # Explanation: imports torch for define the KGCL graph-edit prediction model
import torch.nn as nn  # Explanation: imports torch.nn as nn for define the KGCL graph-edit prediction model
import torch.nn.functional as F  # Explanation: imports torch.nn.functional as F for define the KGCL graph-edit prediction model
from kgcl_retro.chemistry.apply import apply_edit_to_mol  # Explanation: imports the shared packaged edit-application helper.
from rdkit import Chem  # Explanation: imports selected names needed to define the KGCL graph-edit prediction model
from kgcl_retro.data.collate import get_batch_graphs  # Explanation: imports packaged graph batching for autoregressive prediction.
from kgcl_retro.chemistry.graphs import MolGraph, Vocab  # Explanation: imports packaged graph and vocabulary helpers.

from kgcl_retro.models.encoder import Global_Attention, MPNEncoder  # Explanation: imports packaged D-MPNN encoder and optional global attention.
from kgcl_retro.models.utils import (creat_edits_feats, index_select_ND,  # Explanation: imports packaged tensor utilities for edit scoring.
                                     unbatch_feats)  # Explanation: completes the packaged model utility import list.


class KGCL(nn.Module):  # Explanation: defines KGCL, main KGCL neural network for graph-edit retrosynthesis
    def __init__(self,  # Explanation: defines __init__, which define the KGCL graph-edit prediction model
                 config: Dict,  # Explanation: continues the current multi-line argument or data structure
                 atom_vocab: Vocab,  # Explanation: continues the current multi-line argument or data structure
                 bond_vocab: Vocab,  # Explanation: continues the current multi-line argument or data structure
                 device: str = 'cpu') -> None:  # Explanation: assigns an intermediate value used by later computation
        """
        Parameters
        ----------
        config: Dict, Model arguments
        atom_vocab: atom and LG edit labels
        bond_vocab: bond edit labels
        device: str, Device to run the model on.
        """
        super(KGCL, self).__init__()  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model

        self.config = config  # Explanation: stores this value on the object for later model operations
        self.atom_vocab = atom_vocab  # Explanation: stores this value on the object for later model operations
        self.bond_vocab = bond_vocab  # Explanation: stores this value on the object for later model operations
        self.atom_outdim = len(atom_vocab)  # Explanation: stores this value on the object for later model operations
        self.bond_outdim = len(bond_vocab)  # Explanation: stores this value on the object for later model operations
        self.device = device  # Explanation: stores this value on the object for later model operations

        self._build_layers()  # Explanation: uses or updates this object state during computation

    def _build_layers(self) -> None:  # Explanation: defines _build_layers, which define the KGCL graph-edit prediction model
        """Builds the different layers associated with the model."""
        config = self.config  # Explanation: assigns an intermediate value used by later computation
        self.encoder = MPNEncoder(atom_fdim=config['n_atom_feat'],  # Explanation: stores this value on the object for later model operations
                                  bond_fdim=config['n_bond_feat'],  # Explanation: computes an intermediate value for molecular graph editing
                                  hidden_size=config['mpn_size'],  # Explanation: assigns an intermediate value used by later computation
                                  depth=config['depth'],  # Explanation: assigns an intermediate value used by later computation
                                  dropout=config['dropout_mpn'],  # Explanation: assigns an intermediate value used by later computation
                                  atom_message=config['atom_message'])  # Explanation: computes an intermediate value for molecular graph editing

        self.W_vv = nn.Linear(config['mpn_size'],  # Explanation: creates a learned linear projection
                              config['mpn_size'], bias=False)  # Explanation: assigns an intermediate value used by later computation
        nn.init.eye_(self.W_vv.weight)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
        self.W_vc = nn.Linear(config['mpn_size'],  # Explanation: creates a learned linear projection
                              config['mpn_size'], bias=False)  # Explanation: assigns an intermediate value used by later computation

        if config['use_attn']:  # Explanation: checks this condition to choose the next execution path
            self.attn = Global_Attention(  # Explanation: stores this value on the object for later model operations
                d_model=config['mpn_size'], heads=config['n_heads'])  # Explanation: assigns an intermediate value used by later computation

        self.atom_linear = nn.Sequential(  # Explanation: stores this value on the object for later model operations
            nn.Linear(config['mpn_size'], config['mlp_size']),  # Explanation: creates a learned linear projection
            nn.SELU(),  # Explanation: adds SELU nonlinearity
            nn.Dropout(p=config['dropout_mlp']),  # Explanation: adds dropout regularization
            nn.Linear(config['mlp_size'], self.atom_outdim))  # Explanation: creates a learned linear projection
        self.bond_linear = nn.Sequential(  # Explanation: stores this value on the object for later model operations
            nn.Linear(config['mpn_size'] * 2, config['mlp_size']),  # Explanation: creates a learned linear projection
            nn.SELU(),  # Explanation: adds SELU nonlinearity
            nn.Dropout(p=config['dropout_mlp']),  # Explanation: adds dropout regularization
            nn.Linear(config['mlp_size'], self.bond_outdim))  # Explanation: creates a learned linear projection

        self.graph_linear = nn.Sequential(  # Explanation: stores this value on the object for later model operations
            nn.Linear(config['mpn_size'], config['mlp_size']),  # Explanation: creates a learned linear projection
            nn.SELU(),  # Explanation: adds SELU nonlinearity
            nn.Dropout(p=config['dropout_mlp']),  # Explanation: adds dropout regularization
            nn.Linear(config['mlp_size'], 1))  # Explanation: creates a learned linear projection

    def to_device(self, tensors: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:  # Explanation: defines to_device, which define the KGCL graph-edit prediction model
        """Converts all inputs to the device used.

        Parameters
        ----------
        tensors: Union[List, torch.Tensor],
            Tensors to convert to model device. The tensors can be either a
            single tensor or an iterable of tensors.
        """
        if isinstance(tensors, list) or isinstance(tensors, tuple):  # Explanation: checks this condition to choose the next execution path
            tensors = [tensor.to(self.device, non_blocking=True)  # Explanation: assigns an intermediate value used by later computation
                       for tensor in tensors]  # Explanation: iterates over this collection to process each item
            return tensors  # Explanation: returns this computed result to the caller
        elif isinstance(tensors, torch.Tensor):  # Explanation: checks an alternate condition after the previous branch failed
            return tensors.to(self.device, non_blocking=True)  # Explanation: returns this computed result to the caller
        else:  # Explanation: handles the fallback branch for the preceding condition
            raise ValueError(f"Tensors of type {type(tensors)} unsupported")  # Explanation: raises an error when unsupported input is encountered

    def compute_edit_scores(self, prod_tensors: Tuple[torch.Tensor],  # Explanation: defines compute_edit_scores, which define the KGCL graph-edit prediction model
                            prod_scopes: Tuple[List], prev_atom_hiddens: torch.Tensor = None,  # Explanation: computes an intermediate value for molecular graph editing
                            prev_atom_scope: Tuple[List] = None) :  # Explanation: assigns an intermediate value used by later computation
        """Computes the edit scores given product tensors and scopes.

        Parameters
        ----------
        prod_tensors: Tuple[torch.Tensor]:
            Product tensors
        prod_scopes: Tuple[List]
            Product scopes. Scopes is composed of atom and bond scopes, which
            keep track of atom and bond indices for each molecule in the 2D
            feature list
        prev_atom_hiddens: torch.Tensor, default None,
            Previous hidden state of atoms.
        """
        prod_tensors = self.to_device(prod_tensors)  # Explanation: computes an intermediate value for molecular graph editing
        atom_scope, bond_scope = prod_scopes  # Explanation: computes an intermediate value for molecular graph editing
        if prev_atom_hiddens is None:  # Explanation: checks this condition to choose the next execution path
            n_atoms = prod_tensors[0].size(0)  # Explanation: assigns an intermediate value used by later computation
            prev_atom_hiddens = torch.zeros(  # Explanation: assigns an intermediate value used by later computation
                n_atoms, self.config['mpn_size'], device=self.device)  # Explanation: assigns an intermediate value used by later computation

        a_feats = self.encoder(prod_tensors, mask=None)  # Explanation: assigns an intermediate value used by later computation
        if self.config['use_attn']:  # Explanation: checks this condition to choose the next execution path
            feats, mask = creat_edits_feats(a_feats, atom_scope)  # Explanation: assigns an intermediate value used by later computation
            attention_score, feats = self.attn(feats, mask)  # Explanation: assigns an intermediate value used by later computation
            a_feats = unbatch_feats(feats, atom_scope)  # Explanation: assigns an intermediate value used by later computation

        if a_feats.shape[0] != prev_atom_hiddens.shape[0]:  # Explanation: checks this condition to choose the next execution path
            n_atoms = a_feats.shape[0]  # Explanation: assigns an intermediate value used by later computation
            new_ha = torch.zeros(  # Explanation: assigns an intermediate value used by later computation
                n_atoms, self.config['mpn_size'], device=self.device)  # Explanation: assigns an intermediate value used by later computation
            for idx, ((st_n, le_n), (st_p, le_p)) in enumerate(zip(*(atom_scope, prev_atom_scope))):  # Explanation: iterates over this collection to process each item
                new_ha[st_n: st_n + le_p] = prev_atom_hiddens[st_p: st_p + le_p]  # Explanation: assigns an intermediate value used by later computation
            prev_atom_hiddens = new_ha  # Explanation: assigns an intermediate value used by later computation

        assert a_feats.shape == prev_atom_hiddens.shape  # Explanation: checks an invariant expected by the model pipeline
        atom_feats = F.selu(self.W_vv(prev_atom_hiddens) + self.W_vc(a_feats))  # Explanation: computes an intermediate value for molecular graph editing
        prev_atom_hiddens = atom_feats.clone()  # Explanation: assigns an intermediate value used by later computation
        prev_atom_scope = atom_scope  # Explanation: assigns an intermediate value used by later computation

        node_feats = atom_feats.clone()  # Explanation: assigns an intermediate value used by later computation
        bond_starts = index_select_ND(atom_feats, index=prod_tensors[-1][:, 0])  # Explanation: computes an intermediate value for molecular graph editing
        bond_ends = index_select_ND(atom_feats, index=prod_tensors[-1][:, 1])  # Explanation: computes an intermediate value for molecular graph editing
        bond_feats = torch.cat([bond_starts, bond_ends], dim=1)  # Explanation: concatenates tensors along an existing dimension

        graph_vecs = torch.stack(  # Explanation: stacks tensors along a new dimension
            [atom_feats[st: st + le].sum(dim=0) for st, le in atom_scope])  # Explanation: assigns an intermediate value used by later computation

        atom_outs = self.atom_linear(node_feats)  # Explanation: computes an intermediate value for molecular graph editing
        bond_outs = self.bond_linear(bond_feats)  # Explanation: computes an intermediate value for molecular graph editing
        graph_outs = self.graph_linear(graph_vecs)  # Explanation: computes an intermediate value for molecular graph editing

        edit_scores = [torch.cat([bond_outs[st_b: st_b + le_b].flatten(),  # Explanation: concatenates tensors along an existing dimension
                                  atom_outs[st_a: st_a + le_a].flatten(), graph_outs[idx]], dim=-1)  # Explanation: computes an intermediate value for molecular graph editing
                       for idx, ((st_a, le_a), (st_b, le_b)) in enumerate(zip(*(atom_scope, bond_scope)))]  # Explanation: iterates over this collection to process each item

        return edit_scores, prev_atom_hiddens, prev_atom_scope, graph_vecs  # Explanation: returns this computed result to the caller

    def forward(self, prod_seq_inputs: List[Tuple[torch.Tensor, List]]):  # Explanation: defines forward, which define the KGCL graph-edit prediction model
        """
        Forward propagation step.

        Parameters
        ----------
        prod_seq_inputs: List[Tuple[torch.Tensor, List]]
            List of prod_tensors for edit sequence
        """
        max_seq_len = len(prod_seq_inputs)  # Explanation: assigns an intermediate value used by later computation
        assert len(prod_seq_inputs[0]) == 2  # Explanation: checks an invariant expected by the model pipeline

        prev_atom_hiddens = None  # Explanation: assigns an intermediate value used by later computation
        prev_atom_scope = None  # Explanation: assigns an intermediate value used by later computation
        seq_edit_scores = []  # Explanation: computes an intermediate value for molecular graph editing
        batch_graph_outs = []  # Explanation: assigns an intermediate value used by later computation
        for idx in range(max_seq_len):  # Explanation: iterates over this collection to process each item
            prod_tensors, prod_scopes = prod_seq_inputs[idx]  # Explanation: computes an intermediate value for molecular graph editing
            edit_scores, prev_atom_hiddens, prev_atom_scope, graph_outs = self.compute_edit_scores(  # Explanation: computes an intermediate value for molecular graph editing
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
            seq_edit_scores.append(edit_scores)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
            batch_graph_outs.append(graph_outs)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model

        return seq_edit_scores, batch_graph_outs  # Explanation: returns this computed result to the caller

    def predict(self, prod_smi: str, rxn_class: int = None, max_steps: int = 9):  # Explanation: defines predict, which define the KGCL graph-edit prediction model
        """Make predictions for given product smiles string.

        Parameters
        ----------
        prod_smi: str,
            Product SMILES string
        rxn_class: int, default None
            Associated reaction class for the product
        max_steps: int, default 8
            Max number of edit steps allowed
        """
        use_rxn_class = False  # Explanation: assigns an intermediate value used by later computation
        if rxn_class is not None:  # Explanation: checks this condition to choose the next execution path
            use_rxn_class = True  # Explanation: assigns an intermediate value used by later computation

        done = False  # Explanation: assigns an intermediate value used by later computation
        steps = 0  # Explanation: assigns an intermediate value used by later computation
        edits = []  # Explanation: assigns an intermediate value used by later computation
        edits_atom = []  # Explanation: assigns an intermediate value used by later computation
        prev_atom_hiddens = None  # Explanation: assigns an intermediate value used by later computation
        prev_atom_scope = None  # Explanation: assigns an intermediate value used by later computation

        products = Chem.MolFromSmiles(prod_smi)  # Explanation: parses a SMILES string into an RDKit molecule
        Chem.Kekulize(products)  # Explanation: converts aromatic bonds into kekulized form
        prod_graph = MolGraph(mol=Chem.Mol(products),  # Explanation: computes an intermediate value for molecular graph editing
                              rxn_class=rxn_class, use_rxn_class=use_rxn_class)  # Explanation: computes an intermediate value for molecular graph editing
        prod_tensors, prod_scopes = get_batch_graphs(  # Explanation: computes an intermediate value for molecular graph editing
            [prod_graph], use_rxn_class=use_rxn_class)  # Explanation: assigns an intermediate value used by later computation

        while not done and steps <= max_steps:  # Explanation: continues looping while the edit-generation condition remains true
            if prod_tensors[-1].size() == (1, 0):  # Explanation: checks this condition to choose the next execution path
                edit = 'Terminate'  # Explanation: assigns an intermediate value used by later computation
                edits.append(edit)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                done = True  # Explanation: assigns an intermediate value used by later computation
                break  # Explanation: exits the current loop early

            edit_logits, prev_atom_hiddens, prev_atom_scope, graph_outs = self.compute_edit_scores(  # Explanation: computes an intermediate value for molecular graph editing
                prod_tensors, prod_scopes, prev_atom_hiddens, prev_atom_scope)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
            idx = torch.argmax(edit_logits[0])  # Explanation: selects the highest-scoring edit index
            val = edit_logits[0][idx]  # Explanation: assigns an intermediate value used by later computation

            max_bond_idx = products.GetNumBonds() * self.bond_outdim  # Explanation: assigns an intermediate value used by later computation

            if idx.item() == len(edit_logits[0]) - 1:  # Explanation: checks this condition to choose the next execution path
                edit = 'Terminate'  # Explanation: assigns an intermediate value used by later computation
                edits.append(edit)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                done = True  # Explanation: assigns an intermediate value used by later computation
                break  # Explanation: exits the current loop early

            elif idx.item() < max_bond_idx:  # Explanation: checks an alternate condition after the previous branch failed
                bond_logits = edit_logits[0][:products.GetNumBonds(  # Explanation: computes an intermediate value for molecular graph editing
                ) * self.bond_outdim]  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                bond_logits = bond_logits.reshape(  # Explanation: computes an intermediate value for molecular graph editing
                    products.GetNumBonds(), self.bond_outdim)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                idx_tensor = torch.where(bond_logits == val)  # Explanation: assigns an intermediate value used by later computation

                idx_tensor = [indices[-1] for indices in idx_tensor]  # Explanation: assigns an intermediate value used by later computation

                bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()  # Explanation: computes an intermediate value for molecular graph editing
                a1 = products.GetBondWithIdx(  # Explanation: assigns an intermediate value used by later computation
                    bond_idx).GetBeginAtom().GetAtomMapNum()  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                a2 = products.GetBondWithIdx(  # Explanation: assigns an intermediate value used by later computation
                    bond_idx).GetEndAtom().GetAtomMapNum()  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model

                a1, a2 = sorted([a1, a2])  # Explanation: assigns an intermediate value used by later computation
                edit_atom = [a1, a2]  # Explanation: computes an intermediate value for molecular graph editing
                edit = self.bond_vocab.get_elem(edit_idx)  # Explanation: assigns an intermediate value used by later computation

            else:  # Explanation: handles the fallback branch for the preceding condition
                atom_logits = edit_logits[0][max_bond_idx:-1]  # Explanation: computes an intermediate value for molecular graph editing

                assert len(atom_logits) == (  # Explanation: checks that atom logits cover every atom/action pair.
                    products.GetNumAtoms() * self.atom_outdim  # Explanation: computes the expected atom edit logit count.
                )  # Explanation: closes the atom logit shape assertion.
                atom_logits = atom_logits.reshape(  # Explanation: computes an intermediate value for molecular graph editing
                    products.GetNumAtoms(), self.atom_outdim)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                idx_tensor = torch.where(atom_logits == val)  # Explanation: assigns an intermediate value used by later computation

                idx_tensor = [indices[-1] for indices in idx_tensor]  # Explanation: assigns an intermediate value used by later computation
                atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()  # Explanation: computes an intermediate value for molecular graph editing

                a1 = products.GetAtomWithIdx(atom_idx).GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
                edit_atom = a1  # Explanation: computes an intermediate value for molecular graph editing
                edit = self.atom_vocab.get_elem(edit_idx)  # Explanation: assigns an intermediate value used by later computation

            try:  # Explanation: starts a protected block for operations that may fail
                products = apply_edit_to_mol(mol=Chem.Mol(  # Explanation: assigns an intermediate value used by later computation
                    products), edit=edit, edit_atom=edit_atom)  # Explanation: assigns an intermediate value used by later computation
                prod_graph = MolGraph(mol=Chem.Mol(  # Explanation: computes an intermediate value for molecular graph editing
                    products),  rxn_class=rxn_class, use_rxn_class=use_rxn_class)  # Explanation: assigns an intermediate value used by later computation
                prod_tensors, prod_scopes = get_batch_graphs(  # Explanation: computes an intermediate value for molecular graph editing
                    [prod_graph], use_rxn_class=use_rxn_class)  # Explanation: assigns an intermediate value used by later computation

                edits.append(edit)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                edits_atom.append(edit_atom)  # Explanation: executes this statement as part of define the KGCL graph-edit prediction model
                steps += 1  # Explanation: assigns an intermediate value used by later computation

            except:  # Explanation: handles failures from the preceding try block
                steps += 1  # Explanation: assigns an intermediate value used by later computation
                continue  # Explanation: skips the rest of this loop iteration

        return edits, edits_atom  # Explanation: returns this computed result to the caller

    def get_saveables(self) -> Dict:  # Explanation: defines get_saveables, which define the KGCL graph-edit prediction model
        """
        Return the attributes of model used for its construction. This is used
        in restoring the model.
        """
        saveables = {}  # Explanation: assigns an intermediate value used by later computation
        saveables['config'] = self.config  # Explanation: assigns an intermediate value used by later computation
        saveables['atom_vocab'] = self.atom_vocab  # Explanation: assigns an intermediate value used by later computation
        saveables['bond_vocab'] = self.bond_vocab  # Explanation: assigns an intermediate value used by later computation

        return saveables  # Explanation: returns this computed result to the caller
