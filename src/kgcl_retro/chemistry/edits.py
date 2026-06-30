from kgcl_retro.chemistry.actions import (AddGroupAction, AtomEditAction,  # Explanation: imports packaged edit action classes used to label reaction differences.
                                          BondEditAction, Termination)  # Explanation: continues the packaged edit action import list.
from kgcl_retro.chemistry.chem import align_kekulize_pairs, get_atom_info, get_bond_info  # Explanation: imports packaged chemistry helpers for atom and bond comparison.
from rdkit import Chem  # Explanation: imports selected names needed to derive ground-truth edit sequences from reactions
from collections import namedtuple  # Explanation: imports selected names needed to derive ground-truth edit sequences from reactions
from typing import Tuple  # Explanation: imports selected names needed to derive ground-truth edit sequences from reactions

ReactionData = namedtuple(  # Explanation: assigns an intermediate value used by later computation
    "ReactionData", ['rxn_smi', 'edits', 'edits_atom', 'rxn_class', 'rxn_id'])  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions


def generate_reaction_edits(rxn_smi: str, kekulize: bool = False, rxn_class: int = None, rxn_id: str = None) -> ReactionData:  # Explanation: defines generate_reaction_edits, which extracts ground-truth graph edits from a reaction
    # generate bond and atom edits
    r, p = rxn_smi.split(">>")  # Explanation: assigns an intermediate value used by later computation
    react_mol = Chem.MolFromSmiles(r)  # Explanation: parses a SMILES string into an RDKit molecule
    prod_mol = Chem.MolFromSmiles(p)  # Explanation: parses a SMILES string into an RDKit molecule

    if react_mol is None or prod_mol is None:  # Explanation: checks this condition to choose the next execution path
        return None  # Explanation: returns this computed result to the caller

    p_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()  # Explanation: assigns an intermediate value used by later computation
                  for atom in prod_mol.GetAtoms()}  # Explanation: iterates over this collection to process each item

    max_amap = max([atom.GetAtomMapNum() for atom in react_mol.GetAtoms()])  # Explanation: assigns an intermediate value used by later computation
    for atom in react_mol.GetAtoms():  # Explanation: iterates over this collection to process each item
        if atom.GetAtomMapNum() == 0:  # Explanation: checks this condition to choose the next execution path
            atom.SetAtomMapNum(max_amap + 1)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
            max_amap += 1  # Explanation: assigns an intermediate value used by later computation

    r_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()  # Explanation: assigns an intermediate value used by later computation
                  for atom in react_mol.GetAtoms()}  # Explanation: iterates over this collection to process each item

    r_new, p_new = Chem.MolToSmiles(react_mol), Chem.MolToSmiles(prod_mol)  # Explanation: serializes an RDKit molecule back to SMILES
    rxn_smi_new = r_new + ">>" + p_new  # Explanation: computes an intermediate value for molecular graph editing

    if kekulize:  # Explanation: checks this condition to choose the next execution path
        react_mol, prod_mol = align_kekulize_pairs(react_mol, prod_mol)  # Explanation: computes an intermediate value for molecular graph editing

    prod_bonds = get_bond_info(prod_mol)  # Explanation: computes an intermediate value for molecular graph editing
    react_bonds = get_bond_info(react_mol)  # Explanation: computes an intermediate value for molecular graph editing

    edits = []  # Explanation: assigns an intermediate value used by later computation
    edits_atom = []  # Explanation: assigns an intermediate value used by later computation
    bond_edits_atom = set()  # Explanation: computes an intermediate value for molecular graph editing

    for bond in prod_bonds:  # Explanation: iterates over this collection to process each item
        # find delete bonds
        if bond not in react_bonds:  # Explanation: checks this condition to choose the next execution path
            a1, a2 = bond  # Explanation: assigns an intermediate value used by later computation
            edit = BondEditAction(a1, a2, None, None,  # Explanation: assigns an intermediate value used by later computation
                                  action_vocab='Delete Bond')  # Explanation: assigns an intermediate value used by later computation
            edits.append(edit.get_tuple())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
            edits_atom.append([a1, a2])  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
            bond_edits_atom.add(a1)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
            bond_edits_atom.add(a2)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions

    for bond in prod_bonds:  # Explanation: iterates over this collection to process each item
        # find changed bonds
        if bond in react_bonds and prod_bonds[bond] != react_bonds[bond]:  # Explanation: checks this condition to choose the next execution path
            a1, a2 = bond  # Explanation: assigns an intermediate value used by later computation
            edit = BondEditAction(  # Explanation: assigns an intermediate value used by later computation
                a1, a2, *react_bonds[bond], action_vocab='Change Bond')  # Explanation: assigns an intermediate value used by later computation
            edits.append(edit.get_tuple())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
            edits_atom.append([a1, a2])  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
            bond_edits_atom.add(a1)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
            bond_edits_atom.add(a2)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions

    for bond in react_bonds:  # Explanation: iterates over this collection to process each item
        # find new bonds
        if bond not in prod_bonds:  # Explanation: checks this condition to choose the next execution path
            a1, a2 = bond  # Explanation: assigns an intermediate value used by later computation
            if a1 in p_amap_idx and a2 in p_amap_idx:  # Explanation: checks this condition to choose the next execution path
                edit = BondEditAction(  # Explanation: assigns an intermediate value used by later computation
                    a1, a2, *react_bonds[bond], action_vocab='Add Bond')  # Explanation: assigns an intermediate value used by later computation
                edits.append(edit.get_tuple())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                edits_atom.append([a1, a2])  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                bond_edits_atom.add(a1)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                bond_edits_atom.add(a2)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions

    prod_atoms = get_atom_info(prod_mol)  # Explanation: computes an intermediate value for molecular graph editing
    react_atoms = get_atom_info(react_mol)  # Explanation: computes an intermediate value for molecular graph editing
    atoms_only_in_react = []  # Explanation: assigns an intermediate value used by later computation

    for atom in react_atoms:  # Explanation: iterates over this collection to process each item
        if atom not in prod_atoms:  # Explanation: checks this condition to choose the next execution path
            atoms_only_in_react.append(atom)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
        # find changed atoms
    if len(edits_atom) == 0:  # Explanation: checks this condition to choose the next execution path
        for atom in prod_atoms:  # Explanation: iterates over this collection to process each item
            if prod_atoms[atom] != react_atoms[atom]:  # Explanation: checks this condition to choose the next execution path
                edit = AtomEditAction(  # Explanation: assigns an intermediate value used by later computation
                    atom, *react_atoms[atom], action_vocab='Change Atom')  # Explanation: assigns an intermediate value used by later computation
                edits.append(edit.get_tuple())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                edits_atom.append(atom)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
    else:  # Explanation: handles the fallback branch for the preceding condition
        for atom in prod_atoms:  # Explanation: iterates over this collection to process each item
            if prod_atoms[atom] != react_atoms[atom]:  # Explanation: checks this condition to choose the next execution path
                # Exclude edited atoms on bonds
                if atom not in bond_edits_atom:  # Explanation: checks this condition to choose the next execution path
                    edit = AtomEditAction(  # Explanation: assigns an intermediate value used by later computation
                        atom, *react_atoms[atom], action_vocab='Change Atom')  # Explanation: assigns an intermediate value used by later computation
                    edits.append(edit.get_tuple())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                    edits_atom.append(atom)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                # changed atom ChiralTag
                else:  # Explanation: handles the fallback branch for the preceding condition
                    if prod_atoms[atom][1] != react_atoms[atom][1]:  # Explanation: checks this condition to choose the next execution path
                        edit = AtomEditAction(  # Explanation: assigns an intermediate value used by later computation
                            atom, *react_atoms[atom], action_vocab='Change Atom')  # Explanation: assigns an intermediate value used by later computation
                        edits.append(edit.get_tuple())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                        edits_atom.append(atom)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions

    # generate leaving groups
    for bond in react_mol.GetBonds():  # Explanation: iterates over this collection to process each item
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
        a1, a2 = sorted([a1, a2])  # Explanation: assigns an intermediate value used by later computation
        if a1 not in atoms_only_in_react and a2 in atoms_only_in_react:  # Explanation: checks this condition to choose the next execution path
            frags1 = Chem.FragmentOnBonds(react_mol, [bond.GetIdx(  # Explanation: assigns an intermediate value used by later computation
            )], addDummies=True, dummyLabels=[(0, 0)])  # disconnected bond  # Explanation: assigns an intermediate value used by later computation
            frags1_smi = Chem.MolToSmiles(frags1)  # Explanation: serializes an RDKit molecule back to SMILES
            frags1_smi = frags1_smi.split('.')  # Explanation: assigns an intermediate value used by later computation
            for smi in frags1_smi:  # Explanation: iterates over this collection to process each item
                mol = Chem.MolFromSmiles(smi)  # Explanation: parses a SMILES string into an RDKit molecule
                for a in mol.GetAtoms():  # Explanation: iterates over this collection to process each item
                    if a.GetSymbol() == '*':  # Explanation: checks this condition to choose the next execution path
                        atoms_only_in_react.append(a.GetAtomMapNum())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                if all(a.GetAtomMapNum() in atoms_only_in_react for a in mol.GetAtoms()):  # Explanation: checks this condition to choose the next execution path
                    smi = Chem.MolToSmiles(mol)  # Explanation: serializes an RDKit molecule back to SMILES
                    edit = AddGroupAction(a1, smi, action_vocab='Attaching LG')  # Explanation: assigns an intermediate value used by later computation
                    if edit.get_tuple() in edits:  # Explanation: checks this condition to choose the next execution path
                        continue  # Explanation: skips the rest of this loop iteration
                    else:  # Explanation: handles the fallback branch for the preceding condition
                        edits.append(edit.get_tuple())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                        edits_atom.append(a1)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                elif any(a.GetAtomMapNum() in atoms_only_in_react for a in mol.GetAtoms() if a.GetAtomMapNum() != 0):  # Explanation: checks an alternate condition after the previous branch failed
                    for bond in mol.GetBonds():  # Explanation: iterates over this collection to process each item
                        a3, a4 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
                        a3, a4 = sorted([a3, a4])  # Explanation: assigns an intermediate value used by later computation
                        if a3 not in atoms_only_in_react and a3 == a1 and a4 in atoms_only_in_react and a4 != 0:  # Explanation: checks this condition to choose the next execution path
                            frags2 = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[  # Explanation: assigns an intermediate value used by later computation
                                                          (0, 0)])  # disconnected bond  # Explanation: continues a structured literal or expression
                            frags2_smi = Chem.MolToSmiles(frags2)  # Explanation: serializes an RDKit molecule back to SMILES
                            frags2_smi = frags2_smi.split('.')  # Explanation: assigns an intermediate value used by later computation
                            for smi in frags2_smi:  # Explanation: iterates over this collection to process each item
                                mol_2 = Chem.MolFromSmiles(smi)  # Explanation: parses a SMILES string into an RDKit molecule
                                if all(a.GetAtomMapNum() in atoms_only_in_react for a in mol_2.GetAtoms()):  # Explanation: checks this condition to choose the next execution path
                                    smi = Chem.MolToSmiles(mol_2)  # Explanation: serializes an RDKit molecule back to SMILES
                                    edit = AddGroupAction(  # Explanation: assigns an intermediate value used by later computation
                                        a1, smi, action_vocab='Attaching LG')  # Explanation: assigns an intermediate value used by later computation
                                    if edit.get_tuple() in edits:  # Explanation: checks this condition to choose the next execution path
                                        continue  # Explanation: skips the rest of this loop iteration
                                    else:  # Explanation: handles the fallback branch for the preceding condition
                                        edits.append(edit.get_tuple())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
                                        edits_atom.append(a1)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
    # remove lg atom map
    final_edits = []  # Explanation: assigns an intermediate value used by later computation
    for edit in edits:  # Explanation: iterates over this collection to process each item
        if edit[0] == 'Attaching LG':  # Explanation: checks this condition to choose the next execution path
            mol = Chem.MolFromSmiles(edit[1])  # Explanation: parses a SMILES string into an RDKit molecule
            for atom in mol.GetAtoms():  # Explanation: iterates over this collection to process each item
                atom.SetAtomMapNum(0)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions
            smi = Chem.MolToSmiles(mol)  # Explanation: serializes an RDKit molecule back to SMILES
            edit = tuple(['Attaching LG', smi])  # Explanation: assigns an intermediate value used by later computation
        final_edits.append(edit)  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions

    # add stop action finally
    edit = Termination(action_vocab='Terminate')  # Explanation: assigns an intermediate value used by later computation
    final_edits.append(edit.get_tuple())  # Explanation: executes this statement as part of derive ground-truth edit sequences from reactions

    reaction_data = ReactionData(  # Explanation: assigns an intermediate value used by later computation
        rxn_smi=rxn_smi_new, edits=final_edits, edits_atom=edits_atom, rxn_class=rxn_class, rxn_id=rxn_id)  # Explanation: computes an intermediate value for molecular graph editing

    return reaction_data  # Explanation: returns this computed result to the caller
