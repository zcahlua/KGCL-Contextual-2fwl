import numpy as np  # Explanation: imports numpy as np for perform beam-search inference over graph edits
from typing import List  # Explanation: imports selected names needed to perform beam-search inference over graph edits
import torch  # Explanation: imports torch for perform beam-search inference over graph edits
import torch.nn.functional as F  # Explanation: imports torch.nn.functional as F for perform beam-search inference over graph edits
from rdkit import Chem  # Explanation: imports selected names needed to perform beam-search inference over graph edits

from kgcl_retro.chemistry.graphs import MolGraph  # Explanation: imports packaged molecule graph construction for beam states.
from kgcl_retro.data.collate import GraphBatch, get_batch_graphs  # Explanation: imports packaged batching used to score candidate edit states.
from kgcl_retro.chemistry.apply import apply_edit_to_mol  # Explanation: imports the shared edit-application helper used during search.
from kgcl_retro.chemistry.actions import (AddGroupAction, AtomEditAction,  # Explanation: imports packaged action classes used to build beam edit metadata.
                                          BondEditAction, Termination)  # Explanation: completes the packaged edit action import list.


class BeamSearch:  # Explanation: defines BeamSearch, beam-search decoder for edit sequences
    def __init__(self, model, step_beam_size, beam_size, use_rxn_class):  # Explanation: defines __init__, which perform beam-search inference over graph edits
        self.model = model  # Explanation: stores this value on the object for later model operations
        self.step_beam_size = step_beam_size  # Explanation: stores this value on the object for later model operations
        self.beam_size = beam_size  # Explanation: stores this value on the object for later model operations
        self.use_rxn_class = use_rxn_class  # Explanation: stores this value on the object for later model operations
        self.model_variant = getattr(model, "model_variant", "kgcl")

    def _graph_has_no_bonds(self, tensors):
        if isinstance(tensors, GraphBatch):
            return tensors.base_tensors[8].size() == (1, 0)
        return tensors[8].size() == (1, 0)

    def _build_graph(self, mol, rxn_class):
        return MolGraph(
            mol=Chem.Mol(mol),
            rxn_class=rxn_class,
            use_rxn_class=self.use_rxn_class,
            model_variant=self.model_variant,
            use_contextual_fg=getattr(self.model, "config", {}).get("use_contextual_fg", False),
            fg_context_radius=getattr(self.model, "config", {}).get("fg_context_radius", 1),
            fg_max_instances=getattr(self.model, "config", {}).get("fg_max_instances"),
            fg_null_token=getattr(self.model, "config", {}).get("fg_null_token", True),
        )

    def _batch_graph(self, graph):
        config = getattr(self.model, "config", {})
        return get_batch_graphs(
            [graph],
            use_rxn_class=self.use_rxn_class,
            model_variant=self.model_variant,
            fg_context_radius=config.get("fg_context_radius", 1),
            pair_near_radius=config.get("pair_near_radius", 2),
            pair_bridge_radius=config.get("pair_bridge_radius", 2),
            pair_max_score_pairs_enc=config.get("pair_max_score_pairs_enc", 512),
            pair_max_score_pairs_dec=config.get("pair_max_score_pairs_dec", 1024),
            pair_max_carrier_pairs_enc=config.get("pair_max_carrier_pairs_enc", 1024),
            pair_max_carrier_pairs_dec=config.get("pair_max_carrier_pairs_dec", 2048),
            pair_max_bridges_enc=config.get("pair_max_bridges_enc", 8),
            pair_max_bridges_dec=config.get("pair_max_bridges_dec", 8),
            pair_topk=config.get("pair_topk", 64),
        )

    def process_path(self, path, rxn_class):  # Explanation: defines process_path, which perform beam-search inference over graph edits
        new_paths = []  # Explanation: assigns an intermediate value used by later computation

        prod_mol = path['prod_mol']  # Explanation: computes an intermediate value for molecular graph editing
        steps = path['steps'] + 1  # Explanation: assigns an intermediate value used by later computation
        prod_tensors = self.model.to_device(path['tensors'])  # Explanation: computes an intermediate value for molecular graph editing

        if self._graph_has_no_bonds(prod_tensors):  # Explanation: checks this condition to choose the next execution path
            edit = 'Terminate'  # Explanation: assigns an intermediate value used by later computation
            edits_prob, edits = [], []  # Explanation: assigns an intermediate value used by later computation
            edits_prob.extend(path['edits_prob'])  # Explanation: executes this statement as part of perform beam-search inference over graph edits
            edits_prob.append(1.0)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
            edits.extend(path['edits'])  # Explanation: executes this statement as part of perform beam-search inference over graph edits
            edits.append(edit)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
            final_path = {  # Explanation: assigns an intermediate value used by later computation
                'prod_mol': prod_mol,  # Explanation: continues the current multi-line argument or data structure
                'steps': steps,  # Explanation: continues the current multi-line argument or data structure
                'prob': path['prob'],  # Explanation: continues the current multi-line argument or data structure
                'edits_prob': edits_prob,  # Explanation: continues the current multi-line argument or data structure
                'tensors': path['tensors'],  # Explanation: continues the current multi-line argument or data structure
                'scopes': path['scopes'],  # Explanation: continues the current multi-line argument or data structure
                'state': path['state'],  # Explanation: continues the current multi-line argument or data structure
                'state_scope': path['state_scope'],  # Explanation: continues the current multi-line argument or data structure
                'edits': edits,  # Explanation: continues the current multi-line argument or data structure
                'edits_atom': path['edits_atom'],  # Explanation: continues the current multi-line argument or data structure
                'finished': True,  # Explanation: continues the current multi-line argument or data structure
            }  # Explanation: closes the current multi-line expression
            new_paths.append(final_path)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
            return new_paths  # Explanation: returns this computed result to the caller

        edit_logits, state, state_scope, graph_vecs = self.model.compute_edit_scores(  # Explanation: computes an intermediate value for molecular graph editing
                prod_tensors, path['scopes'], path['state'], path['state_scope'])  # Explanation: executes this statement as part of perform beam-search inference over graph edits
        edit_logits = edit_logits[0]  # Explanation: computes an intermediate value for molecular graph editing
        edit_logits = F.softmax(edit_logits, dim=-1)  # Explanation: converts edit logits into probabilities

        k = self.step_beam_size  # Explanation: assigns an intermediate value used by later computation
        top_k_vals, top_k_idxs = torch.topk(edit_logits, k=k)  # Explanation: selects top edit candidates for beam search

        for beam_idx, (topk_idx, val) in enumerate(zip(*(top_k_idxs, top_k_vals))):  # Explanation: iterates over this collection to process each item
            edit, edit_atom = self.get_edit_from_logits(  # Explanation: assigns an intermediate value used by later computation
                mol=prod_mol, edit_logits=edit_logits, idx=topk_idx, val=val, graph_batch=prod_tensors)  # Explanation: assigns an intermediate value used by later computation
            val = round(val.item(), 4)  # Explanation: assigns an intermediate value used by later computation
            new_prob = path['prob'] * val  # Explanation: assigns an intermediate value used by later computation

            if edit == 'Terminate':  # Explanation: checks this condition to choose the next execution path
                edits_prob, edits = [], []  # Explanation: assigns an intermediate value used by later computation
                edits_prob.extend(path['edits_prob'])  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                edits_prob.append(val)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                edits.extend(path['edits'])  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                edits.append(edit)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                final_path = {  # Explanation: assigns an intermediate value used by later computation
                    'prod_mol': prod_mol,  # Explanation: continues the current multi-line argument or data structure
                    'steps': steps,  # Explanation: continues the current multi-line argument or data structure
                    'prob': new_prob,  # Explanation: continues the current multi-line argument or data structure
                    'edits_prob': edits_prob,  # Explanation: continues the current multi-line argument or data structure
                    'tensors': path['tensors'],  # Explanation: continues the current multi-line argument or data structure
                    'scopes': path['scopes'],  # Explanation: continues the current multi-line argument or data structure
                    'state': state,  # Explanation: continues the current multi-line argument or data structure
                    'state_scope': state_scope,  # Explanation: continues the current multi-line argument or data structure
                    'edits': edits,  # Explanation: continues the current multi-line argument or data structure
                    'edits_atom': path['edits_atom'],  # Explanation: continues the current multi-line argument or data structure
                    'finished': True,  # Explanation: continues the current multi-line argument or data structure
                }  # Explanation: closes the current multi-line expression
                new_paths.append(final_path)  # Explanation: executes this statement as part of perform beam-search inference over graph edits

            else:  # Explanation: handles the fallback branch for the preceding condition
                try:  # Explanation: starts a protected block for operations that may fail
                    int_mol = apply_edit_to_mol(mol=Chem.Mol(  # Explanation: assigns an intermediate value used by later computation
                        prod_mol), edit=edit, edit_atom=edit_atom)  # Explanation: computes an intermediate value for molecular graph editing
                    prod_graph = self._build_graph(int_mol, rxn_class)  # Explanation: computes an intermediate value for molecular graph editing
                    graph_batch = self._batch_graph(prod_graph)
                    if isinstance(graph_batch, GraphBatch):
                        prod_tensors, prod_scopes = graph_batch, None
                    else:
                        prod_tensors, prod_scopes = graph_batch  # Explanation: assigns an intermediate value used by later computation
                    edits_prob, edits, edits_atom = [], [], []  # Explanation: assigns an intermediate value used by later computation
                    edits_prob.extend(path['edits_prob'])  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                    edits_prob.append(val)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                    edits.extend(path['edits'])  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                    edits.append(edit)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                    edits_atom.extend(path['edits_atom'])  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                    edits_atom.append(edit_atom)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                    new_path = {  # Explanation: assigns an intermediate value used by later computation
                        'prod_mol': int_mol,  # Explanation: continues the current multi-line argument or data structure
                        'steps': steps,  # Explanation: continues the current multi-line argument or data structure
                        'prob': new_prob,  # Explanation: continues the current multi-line argument or data structure
                        'edits_prob': edits_prob,  # Explanation: continues the current multi-line argument or data structure
                        'tensors': prod_tensors,  # Explanation: continues the current multi-line argument or data structure
                        'scopes': prod_scopes,  # Explanation: continues the current multi-line argument or data structure
                        'state': state,  # Explanation: continues the current multi-line argument or data structure
                        'state_scope': state_scope,  # Explanation: continues the current multi-line argument or data structure
                        'edits': edits,  # Explanation: continues the current multi-line argument or data structure
                        'edits_atom': edits_atom,  # Explanation: continues the current multi-line argument or data structure
                        'finished': False,  # Explanation: continues the current multi-line argument or data structure
                    }  # Explanation: closes the current multi-line expression
                    new_paths.append(new_path)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                except:  # Explanation: handles failures from the preceding try block
                    continue  # Explanation: skips the rest of this loop iteration

        return new_paths  # Explanation: returns this computed result to the caller

    def get_top_k_paths(self, paths):  # Explanation: defines get_top_k_paths, which perform beam-search inference over graph edits
        k = min(len(paths), self.beam_size)  # Explanation: assigns an intermediate value used by later computation
        path_argsort = np.argsort([-path['prob'] for path in paths])  # Explanation: assigns an intermediate value used by later computation
        filtered_paths = [paths[i] for i in path_argsort[:k]]  # Explanation: assigns an intermediate value used by later computation

        return filtered_paths  # Explanation: returns this computed result to the caller

    def get_edit_from_logits(self, mol, edit_logits, idx, val, graph_batch=None):  # Explanation: defines get_edit_from_logits, which perform beam-search inference over graph edits
        if self.model_variant == "contextual_2fwl":
            return self.model.contextual_2fwl.decode_action(mol, graph_batch, edit_logits, idx)

        max_bond_idx = mol.GetNumBonds() * self.model.bond_outdim  # Explanation: assigns an intermediate value used by later computation

        if idx.item() == len(edit_logits) - 1:  # Explanation: checks this condition to choose the next execution path
            edit = 'Terminate'  # Explanation: assigns an intermediate value used by later computation
            edit_atom = []  # Explanation: computes an intermediate value for molecular graph editing

        elif idx.item() < max_bond_idx:  # Explanation: checks an alternate condition after the previous branch failed
            bond_logits = edit_logits[:mol.GetNumBonds(  # Explanation: computes an intermediate value for molecular graph editing
            ) * self.model.bond_outdim]  # Explanation: executes this statement as part of perform beam-search inference over graph edits
            bond_logits = bond_logits.reshape(  # Explanation: computes an intermediate value for molecular graph editing
                mol.GetNumBonds(), self.model.bond_outdim)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
            idx_tensor = torch.where(bond_logits == val)  # Explanation: assigns an intermediate value used by later computation

            idx_tensor = [indices[-1] for indices in idx_tensor]  # Explanation: assigns an intermediate value used by later computation

            bond_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()  # Explanation: computes an intermediate value for molecular graph editing
            a1 = mol.GetBondWithIdx(bond_idx).GetBeginAtom().GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
            a2 = mol.GetBondWithIdx(bond_idx).GetEndAtom().GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation

            a1, a2 = sorted([a1, a2])  # Explanation: assigns an intermediate value used by later computation
            edit_atom = [a1, a2]  # Explanation: computes an intermediate value for molecular graph editing
            edit = self.model.bond_vocab.get_elem(edit_idx)  # Explanation: assigns an intermediate value used by later computation

        else:  # Explanation: handles the fallback branch for the preceding condition
            atom_logits = edit_logits[max_bond_idx:-1]  # Explanation: computes an intermediate value for molecular graph editing

            assert len(atom_logits) == (  # Explanation: checks that atom logits cover every atom/action pair.
                mol.GetNumAtoms() * self.model.atom_outdim  # Explanation: computes the expected atom edit logit count.
            )  # Explanation: closes the atom logit shape assertion.
            atom_logits = atom_logits.reshape(  # Explanation: computes an intermediate value for molecular graph editing
                mol.GetNumAtoms(), self.model.atom_outdim)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
            idx_tensor = torch.where(atom_logits == val)  # Explanation: assigns an intermediate value used by later computation

            idx_tensor = [indices[-1] for indices in idx_tensor]  # Explanation: assigns an intermediate value used by later computation
            atom_idx, edit_idx = idx_tensor[0].item(), idx_tensor[1].item()  # Explanation: computes an intermediate value for molecular graph editing

            a1 = mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
            edit_atom = a1  # Explanation: computes an intermediate value for molecular graph editing
            edit = self.model.atom_vocab.get_elem(edit_idx)  # Explanation: assigns an intermediate value used by later computation

        return edit, edit_atom  # Explanation: returns this computed result to the caller

    def run_search(self, prod_smi: str, max_steps: int = 8, rxn_class: int = None) -> List[dict]:  # Explanation: defines run_search, which perform beam-search inference over graph edits
        product = Chem.MolFromSmiles(prod_smi)  # Explanation: parses a SMILES string into an RDKit molecule
        Chem.Kekulize(product)  # Explanation: converts aromatic bonds into kekulized form
        prod_graph = self._build_graph(product, rxn_class)  # Explanation: computes an intermediate value for molecular graph editing
        graph_batch = self._batch_graph(prod_graph)
        if isinstance(graph_batch, GraphBatch):
            prod_tensors, prod_scopes = graph_batch, None
        else:
            prod_tensors, prod_scopes = graph_batch  # Explanation: assigns an intermediate value used by later computation

        paths = []  # Explanation: assigns an intermediate value used by later computation
        start_path = {  # Explanation: assigns an intermediate value used by later computation
            'prod_mol': product,  # Explanation: continues the current multi-line argument or data structure
            'steps': 0,  # Explanation: continues the current multi-line argument or data structure
            'prob': 1.0,  # Explanation: continues the current multi-line argument or data structure
            'edits_prob': [],  # Explanation: continues the current multi-line argument or data structure
            'tensors': prod_tensors,  # Explanation: continues the current multi-line argument or data structure
            'scopes': prod_scopes,  # Explanation: continues the current multi-line argument or data structure
            'state': None,  # Explanation: continues the current multi-line argument or data structure
            'state_scope': None,  # Explanation: continues the current multi-line argument or data structure
            'edits': [],  # Explanation: continues the current multi-line argument or data structure
            'edits_atom': [],  # Explanation: continues the current multi-line argument or data structure
            'finished': False,  # Explanation: continues the current multi-line argument or data structure
        }  # Explanation: closes the current multi-line expression
        paths.append(start_path)  # Explanation: executes this statement as part of perform beam-search inference over graph edits

        for step_i in range(max_steps):  # Explanation: iterates over this collection to process each item
            followed_path = [path for path in paths if not path['finished']]  # Explanation: assigns an intermediate value used by later computation
            if len(followed_path) == 0:  # Explanation: checks this condition to choose the next execution path
                break  # Explanation: exits the current loop early

            paths = [path for path in paths if path['finished']]  # Explanation: assigns an intermediate value used by later computation

            for path in followed_path:  # Explanation: iterates over this collection to process each item
                new_paths = self.process_path(path, rxn_class)  # Explanation: assigns an intermediate value used by later computation
                paths += new_paths  # Explanation: assigns an intermediate value used by later computation

            paths = self.get_top_k_paths(paths)  # Explanation: assigns an intermediate value used by later computation

            if all(path['finished'] for path in paths):  # Explanation: checks this condition to choose the next execution path
                break  # Explanation: exits the current loop early

        finished_paths = []  # Explanation: assigns an intermediate value used by later computation
        for path in paths:  # Explanation: iterates over this collection to process each item
            if path['finished']:  # Explanation: checks this condition to choose the next execution path
                try:  # Explanation: starts a protected block for operations that may fail
                    int_mol = product  # Explanation: assigns an intermediate value used by later computation
                    path['rxn_actions'] = []  # Explanation: assigns an intermediate value used by later computation
                    for i, edit in enumerate(path['edits']):  # Explanation: iterates over this collection to process each item
                        if int_mol is None:  # Explanation: checks this condition to choose the next execution path
                            print("Interim mol is None")  # Explanation: prints progress or diagnostic information
                            break  # Explanation: exits the current loop early
                        if edit == 'Terminate':  # Explanation: checks this condition to choose the next execution path
                            edit_exe = Termination(action_vocab='Terminate')  # Explanation: computes an intermediate value for molecular graph editing
                            path['rxn_actions'].append(edit_exe)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                            pred_mol = edit_exe.apply(int_mol)  # Explanation: assigns an intermediate value used by later computation
                            [a.ClearProp('molAtomMapNumber')  # Explanation: executes this list-comprehension side effect over molecule atoms
                             for a in pred_mol.GetAtoms()]  # Explanation: iterates over this collection to process each item
                            pred_mol = Chem.MolFromSmiles(  # Explanation: parses a SMILES string into an RDKit molecule
                                Chem.MolToSmiles(pred_mol))  # Explanation: serializes an RDKit molecule back to SMILES
                            final_smi = Chem.MolToSmiles(pred_mol)  # Explanation: serializes an RDKit molecule back to SMILES
                            path['final_smi'] = final_smi  # Explanation: assigns an intermediate value used by later computation

                        elif edit[0] == 'Change Atom':  # Explanation: checks an alternate condition after the previous branch failed
                            edit_exe = AtomEditAction(  # Explanation: computes an intermediate value for molecular graph editing
                                path['edits_atom'][i], *edit[1], action_vocab='Change Atom')  # Explanation: assigns an intermediate value used by later computation
                            path['rxn_actions'].append(edit_exe)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                            int_mol = edit_exe.apply(int_mol)  # Explanation: assigns an intermediate value used by later computation

                        elif edit[0] == 'Delete Bond':  # Explanation: checks an alternate condition after the previous branch failed
                            edit_exe = BondEditAction(  # Explanation: computes an intermediate value for molecular graph editing
                                *path['edits_atom'][i], *edit[1], action_vocab='Delete Bond')  # Explanation: assigns an intermediate value used by later computation
                            path['rxn_actions'].append(edit_exe)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                            int_mol = edit_exe.apply(int_mol)  # Explanation: assigns an intermediate value used by later computation

                        if edit[0] == 'Change Bond':  # Explanation: checks this condition to choose the next execution path
                            edit_exe = BondEditAction(  # Explanation: computes an intermediate value for molecular graph editing
                                *path['edits_atom'][i], *edit[1], action_vocab='Change Bond')  # Explanation: assigns an intermediate value used by later computation
                            path['rxn_actions'].append(edit_exe)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                            int_mol = edit_exe.apply(int_mol)  # Explanation: assigns an intermediate value used by later computation

                        if edit[0] == 'Attaching LG':  # Explanation: checks this condition to choose the next execution path
                            edit_exe = AddGroupAction(  # Explanation: computes an intermediate value for molecular graph editing
                                path['edits_atom'][i], edit[1], action_vocab='Attaching LG')  # Explanation: assigns an intermediate value used by later computation
                            path['rxn_actions'].append(edit_exe)  # Explanation: executes this statement as part of perform beam-search inference over graph edits
                            int_mol = edit_exe.apply(int_mol)  # Explanation: assigns an intermediate value used by later computation

                    finished_paths.append(path)  # Explanation: executes this statement as part of perform beam-search inference over graph edits

                except Exception as e:  # Explanation: handles failures from the preceding try block
                    print(f'Exception while final mol to Smiles: {str(e)}')  # Explanation: prints progress or diagnostic information
                    path['final_smi'] = 'final_smi_unmapped'  # Explanation: assigns an intermediate value used by later computation
                    finished_paths.append(path)  # Explanation: executes this statement as part of perform beam-search inference over graph edits

        return finished_paths  # Explanation: returns this computed result to the caller
