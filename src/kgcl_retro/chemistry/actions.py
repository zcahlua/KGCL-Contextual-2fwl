"""
Definitions of basic 'edits' (Actions) to transform a product into synthons and reactants
"""
from abc import ABCMeta, abstractmethod  # Explanation: imports selected names needed to define executable molecular graph edit actions
from typing import Optional, Tuple  # Explanation: imports selected names needed to define executable molecular graph edit actions

from rdkit import Chem  # Explanation: imports selected names needed to define executable molecular graph edit actions
from rdkit.Chem import Mol, rdchem  # Explanation: imports selected names needed to define executable molecular graph edit actions

from kgcl_retro.chemistry.chem import attach_lg, fix_Hs_Charge, get_atom_Chiral, get_bond_stereo  # Explanation: imports chemistry helpers from the package instead of the legacy utils path.

MAX_BONDS = {'C': 4, 'N': 3, 'O': 2, 'Br': 1, 'Cl': 1, 'F': 1, 'I': 1}  # Explanation: defines a module-level constant used by the pipeline


class ReactionAction(metaclass=ABCMeta):  # Explanation: defines ReactionAction, abstract base class for molecular edit actions
    def __init__(self, atom_map1: int, atom_map2: int, action_vocab: str):  # Explanation: defines __init__, which define executable molecular graph edit actions
        self.atom_map1 = atom_map1  # Explanation: stores this value on the object for later model operations
        self.atom_map2 = atom_map2  # Explanation: stores this value on the object for later model operations
        self.action_vocab = action_vocab  # Explanation: stores this value on the object for later model operations

    @abstractmethod  # Explanation: applies this decorator to the following method or function
    def get_tuple(self) -> Tuple[str, ...]:  # Explanation: defines get_tuple, which define executable molecular graph edit actions
        raise NotImplementedError('Abstract method')  # Explanation: raises an error when unsupported input is encountered

    @abstractmethod  # Explanation: applies this decorator to the following method or function
    def apply(self, mol: Mol) -> Mol:  # Explanation: defines apply, which define executable molecular graph edit actions
        raise NotImplementedError('Abstract method')  # Explanation: raises an error when unsupported input is encountered


class AtomEditAction(ReactionAction):  # Explanation: defines AtomEditAction, edit action that changes atom features
    def __init__(self, atom_map1: int, num_explicit_hs: int, chiral_tag: int, action_vocab: str):  # Explanation: defines __init__, which define executable molecular graph edit actions
        super(AtomEditAction, self).__init__(atom_map1, -1, action_vocab)  # Explanation: executes this statement as part of define executable molecular graph edit actions
        self.num_explicit_hs = num_explicit_hs  # Explanation: stores this value on the object for later model operations
        self.chiral_tag = chiral_tag  # Explanation: stores this value on the object for later model operations

    @property  # Explanation: applies this decorator to the following method or function
    def feat_vals(self) -> Tuple[int, int]:  # Explanation: defines feat_vals, which define executable molecular graph edit actions
        return self.num_explicit_hs, self.chiral_tag  # Explanation: returns this computed result to the caller

    def get_tuple(self) -> Tuple[str, Tuple[int, int]]:  # Explanation: defines get_tuple, which define executable molecular graph edit actions
        return self.action_vocab, self.feat_vals  # Explanation: returns this computed result to the caller

    def apply(self, mol: Mol) -> Mol:  # Explanation: defines apply, which define executable molecular graph edit actions
        new_mol = Chem.RWMol(mol)  # Explanation: creates an editable RDKit molecule
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()  # Explanation: assigns an intermediate value used by later computation
                    if atom.GetAtomMapNum() != 0}  # Explanation: checks this condition to choose the next execution path
        atom_idx = amap_idx[self.atom_map1]  # Explanation: computes an intermediate value for molecular graph editing
        atom = new_mol.GetAtomWithIdx(atom_idx)  # Explanation: assigns an intermediate value used by later computation
        atom.SetNumExplicitHs(self.num_explicit_hs)  # Explanation: executes this statement as part of define executable molecular graph edit actions
        a_chiral = rdchem.ChiralType.values[self.chiral_tag]  # Explanation: assigns an intermediate value used by later computation
        atom.SetChiralTag(a_chiral)  # Explanation: executes this statement as part of define executable molecular graph edit actions
        pred_mol = new_mol.GetMol()  # Explanation: assigns an intermediate value used by later computation
        return pred_mol  # Explanation: returns this computed result to the caller

    def __str__(self):  # Explanation: defines __str__, which define executable molecular graph edit actions
        return f'Edit Atom {self.atom_map1}: Num explicit Hs={self.num_explicit_hs}, Chiral_tag={self.chiral_tag}'  # Explanation: returns this computed result to the caller


class BondEditAction(ReactionAction):  # Explanation: defines BondEditAction, edit action that changes or deletes bonds
    def __init__(self, atom_map1: int, atom_map2: int,  # Explanation: defines __init__, which define executable molecular graph edit actions
                 bond_type: Optional[int], bond_stereo: Optional[int],  # Explanation: continues the current multi-line argument or data structure
                 action_vocab: str):  # Explanation: executes this statement as part of define executable molecular graph edit actions
        super(BondEditAction, self).__init__(  # Explanation: executes this statement as part of define executable molecular graph edit actions
            atom_map1, atom_map2, action_vocab)  # Explanation: executes this statement as part of define executable molecular graph edit actions
        self.bond_type = bond_type  # Explanation: stores this value on the object for later model operations
        self.bond_stereo = bond_stereo  # Explanation: stores this value on the object for later model operations

    def get_tuple(self) -> Tuple[str, Tuple[Optional[int], Optional[int]]]:  # Explanation: defines get_tuple, which define executable molecular graph edit actions
        return self.action_vocab, (self.bond_type, self.bond_stereo)  # Explanation: returns this computed result to the caller

    def apply(self, mol: Mol) -> Mol:  # Explanation: defines apply, which define executable molecular graph edit actions
        new_mol = Chem.RWMol(mol)  # Explanation: creates an editable RDKit molecule
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()  # Explanation: assigns an intermediate value used by later computation
                    if atom.GetAtomMapNum() != 0}  # Explanation: checks this condition to choose the next execution path
        atom1 = new_mol.GetAtomWithIdx(amap_idx[self.atom_map1])  # Explanation: assigns an intermediate value used by later computation
        atom2 = new_mol.GetAtomWithIdx(amap_idx[self.atom_map2])  # Explanation: assigns an intermediate value used by later computation

        if self.bond_type is None:  # delete bond  # Explanation: checks this condition to choose the next execution path
            bond = new_mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())  # Explanation: assigns an intermediate value used by later computation
            new_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())  # Explanation: executes this statement as part of define executable molecular graph edit actions
            pred_mol = new_mol.GetMol()  # Explanation: assigns an intermediate value used by later computation

        else:  # Explanation: handles the fallback branch for the preceding condition
            b_type = rdchem.BondType.values[self.bond_type]  # Explanation: assigns an intermediate value used by later computation
            b_stereo = rdchem.BondStereo.values[self.bond_stereo]  # Explanation: assigns an intermediate value used by later computation

            bond = new_mol.GetBondBetweenAtoms(atom1.GetIdx(), atom2.GetIdx())  # Explanation: assigns an intermediate value used by later computation
            b1 = bond.GetBondTypeAsDouble()  # Explanation: assigns an intermediate value used by later computation
            if bond is None:  # add new bond  # Explanation: checks this condition to choose the next execution path
                pass  # Explanation: leaves this branch intentionally empty
            else:  # change an existing bond  # Explanation: handles the fallback branch for the preceding condition
                bond.SetBondType(b_type)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                bond.SetStereo(b_stereo)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                b2 = bond.GetBondTypeAsDouble()  # Explanation: assigns an intermediate value used by later computation

                val = b1 - b2  # Explanation: assigns an intermediate value used by later computation
                if val > 0:  # Explanation: checks this condition to choose the next execution path
                    atom1.SetNumExplicitHs(int(atom1.GetNumExplicitHs() + val))  # Explanation: executes this statement as part of define executable molecular graph edit actions
                    atom2.SetNumExplicitHs(int(atom2.GetNumExplicitHs() + val))  # Explanation: executes this statement as part of define executable molecular graph edit actions

                elif val < 0:  # Explanation: checks an alternate condition after the previous branch failed
                    atom1.SetNumExplicitHs(  # Explanation: executes this statement as part of define executable molecular graph edit actions
                        int(max(0, atom1.GetNumExplicitHs() + val)))  # Explanation: executes this statement as part of define executable molecular graph edit actions
                    atom2.SetNumExplicitHs(  # Explanation: executes this statement as part of define executable molecular graph edit actions
                        int(max(0, atom2.GetNumExplicitHs() + val)))  # Explanation: executes this statement as part of define executable molecular graph edit actions

                if atom1.GetSymbol() == 'S' and atom2.GetSymbol() == 'O':  # Explanation: checks this condition to choose the next execution path
                    if b1 == 1.0 and b2 == 2.0 and atom2.GetFormalCharge() == -1:  # Explanation: checks this condition to choose the next execution path
                        atom2.SetFormalCharge(0)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                    if b1 == 2.0 and b2 == 1.0 and atom2.GetFormalCharge() == 0:  # Explanation: checks this condition to choose the next execution path
                        atom1.SetNumExplicitHs(0)  # Explanation: executes this statement as part of define executable molecular graph edit actions

                elif atom2.GetSymbol() == 'S' and atom1.GetSymbol() == 'O':  # Explanation: checks an alternate condition after the previous branch failed
                    if b1 == 1.0 and b2 == 2.0 and atom1.GetFormalCharge() == -1:  # Explanation: checks this condition to choose the next execution path
                        atom1.SetFormalCharge(0)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                    if b1 == 2.0 and b2 == 1.0 and atom1.GetFormalCharge() == 0:  # Explanation: checks this condition to choose the next execution path
                        atom2.SetNumExplicitHs(0)  # Explanation: executes this statement as part of define executable molecular graph edit actions

                elif atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'N':  # Explanation: checks an alternate condition after the previous branch failed
                    if b1 == 3.0 and b2 == 1.0 and atom1.GetFormalCharge() == -1 and atom2.GetFormalCharge() == 1:  # Explanation: checks this condition to choose the next execution path
                        atom1.SetFormalCharge(0)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                        atom2.SetFormalCharge(0)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                        atom1.SetNumExplicitHs(1)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                        atom2.SetNumExplicitHs(1)  # Explanation: executes this statement as part of define executable molecular graph edit actions

                elif atom2.GetSymbol() == 'C' and atom1.GetSymbol() == 'N':  # Explanation: checks an alternate condition after the previous branch failed
                    if b1 == 3.0 and b2 == 1.0 and atom2.GetFormalCharge() == -1 and atom1.GetFormalCharge() == 1:  # Explanation: checks this condition to choose the next execution path
                        atom1.SetFormalCharge(0)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                        atom2.SetFormalCharge(0)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                        atom1.SetNumExplicitHs(1)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                        atom2.SetNumExplicitHs(1)  # Explanation: executes this statement as part of define executable molecular graph edit actions

            pred_mol = new_mol.GetMol()  # Explanation: assigns an intermediate value used by later computation
            # fix explicit Hs and charge
            pred_mol = fix_Hs_Charge(pred_mol)  # Explanation: assigns an intermediate value used by later computation

        return pred_mol  # Explanation: returns this computed result to the caller

    def __str__(self):  # Explanation: defines __str__, which define executable molecular graph edit actions
        if self.bond_type is None:  # Explanation: checks this condition to choose the next execution path
            return f'Delete bond {self.atom_map1, self.atom_map2}'  # Explanation: returns this computed result to the caller
        bond_feat = f'Bond type={self.bond_type}, Bond Stereo={self.bond_stereo}'  # Explanation: computes an intermediate value for molecular graph editing
        return f'{self.action_vocab} {self.atom_map1, self.atom_map2}: {bond_feat}'  # Explanation: returns this computed result to the caller


class AddGroupAction(ReactionAction):  # Explanation: defines AddGroupAction, edit action that attaches a leaving group
    def __init__(self, atom_map1: int, leaving_group: str, action_vocab: str):  # Explanation: defines __init__, which define executable molecular graph edit actions
        super(AddGroupAction, self).__init__(atom_map1, -1, action_vocab)  # Explanation: executes this statement as part of define executable molecular graph edit actions
        self.leaving_group = leaving_group  # Explanation: stores this value on the object for later model operations

    def get_tuple(self) -> Tuple[str, str]:  # Explanation: defines get_tuple, which define executable molecular graph edit actions
        return self.action_vocab, self.leaving_group  # Explanation: returns this computed result to the caller

    def apply(self, mol: Mol) -> Mol:  # Explanation: defines apply, which define executable molecular graph edit actions
        lg_mol = Chem.MolFromSmiles(self.leaving_group)  # Explanation: parses a SMILES string into an RDKit molecule
        Chem.Kekulize(lg_mol)  # Explanation: converts aromatic bonds into kekulized form
        try:  # Explanation: starts a protected block for operations that may fail
            pred_mol = attach_lg(main_mol=mol, lg_mol=lg_mol,  # Explanation: assigns an intermediate value used by later computation
                                 attach_atom_map=self.atom_map1)  # Explanation: assigns an intermediate value used by later computation
        except Exception as e:  # Explanation: handles failures from the preceding try block
            print('fail to attach lg')  # Explanation: prints progress or diagnostic information
            pred_mol = mol  # Explanation: assigns an intermediate value used by later computation
        # fix explicit Hs and charge
        pred_mol = fix_Hs_Charge(pred_mol)  # Explanation: assigns an intermediate value used by later computation
        return pred_mol  # Explanation: returns this computed result to the caller

    def __str__(self):  # Explanation: defines __str__, which define executable molecular graph edit actions
        return f'Attaching {self.leaving_group} to atom {self.atom_map1}'  # Explanation: returns this computed result to the caller


class Termination(ReactionAction):  # Explanation: defines Termination, edit action that finalizes reactants
    def __init__(self, action_vocab: str):  # Explanation: defines __init__, which define executable molecular graph edit actions
        super(Termination, self).__init__(-1, -1, action_vocab=action_vocab)  # Explanation: assigns an intermediate value used by later computation

    def get_tuple(self) -> Tuple[str]:  # Explanation: defines get_tuple, which define executable molecular graph edit actions
        return self.action_vocab  # Explanation: returns this computed result to the caller

    def apply(self, mol: Mol) -> Mol:  # Explanation: defines apply, which define executable molecular graph edit actions

        atom_chiral = get_atom_Chiral(mol)  # Explanation: computes an intermediate value for molecular graph editing
        bond_stereo = get_bond_stereo(mol)  # Explanation: computes an intermediate value for molecular graph editing
        if all(int(bt) == 0 for bt in bond_stereo.values()) and all(int(chiral) == 0 for chiral in atom_chiral.values()):  # Explanation: checks this condition to choose the next execution path
            mol = Chem.MolFromSmiles(  # Explanation: parses a SMILES string into an RDKit molecule
                Chem.MolToSmiles(mol, isomericSmiles=False))  # Explanation: serializes an RDKit molecule back to SMILES
            return mol  # Explanation: returns this computed result to the caller

        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # Explanation: parses a SMILES string into an RDKit molecule

        # Dealing with the inconsistency between molecular mol chirality and atomic chiral tag
        for atom in mol.GetAtoms():  # Explanation: iterates over this collection to process each item
            amap_num = atom.GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
            atom.SetChiralTag(atom_chiral[amap_num])  # Explanation: executes this statement as part of define executable molecular graph edit actions

        # Dealing with the inconsistency between molecular mol stereo and bond stereo
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()  # Explanation: assigns an intermediate value used by later computation
                    for atom in mol.GetAtoms()}  # Explanation: iterates over this collection to process each item
        for bond in mol.GetBonds():  # Explanation: iterates over this collection to process each item
            a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
            atom1 = mol.GetAtomWithIdx(amap_idx[a1])  # Explanation: assigns an intermediate value used by later computation
            atom2 = mol.GetAtomWithIdx(amap_idx[a2])  # Explanation: assigns an intermediate value used by later computation
            bond_atoms = sorted([a1, a2])  # Explanation: computes an intermediate value for molecular graph editing
            st = bond_stereo[tuple(bond_atoms)]  # Explanation: assigns an intermediate value used by later computation

            a1_max_neigh = None  # Explanation: assigns an intermediate value used by later computation
            a2_max_neigh = None  # Explanation: assigns an intermediate value used by later computation
            bond.SetStereo(st)  # Explanation: executes this statement as part of define executable molecular graph edit actions

            if int(st) != 0 and len(list(bond.GetStereoAtoms())) < 2:  # Explanation: checks this condition to choose the next execution path
                if len([a.GetAtomicNum() for a in atom1.GetNeighbors() if a.GetIdx() != atom2.GetIdx()]) == 0 or len(  # Explanation: checks this condition to choose the next execution path
                        [a.GetAtomicNum() for a in atom2.GetNeighbors() if a.GetIdx() != atom1.GetIdx()]) == 0:  # Explanation: assigns an intermediate value used by later computation
                    continue  # Explanation: skips the rest of this loop iteration
                else:  # Explanation: handles the fallback branch for the preceding condition
                    a1_max_neigh_num = max(  # Explanation: assigns an intermediate value used by later computation
                        [a.GetAtomicNum() for a in atom1.GetNeighbors() if a.GetIdx() != atom2.GetIdx()])  # Explanation: assigns an intermediate value used by later computation
                    a2_max_neigh_num = max(  # Explanation: assigns an intermediate value used by later computation
                        [a.GetAtomicNum() for a in atom2.GetNeighbors() if a.GetIdx() != atom1.GetIdx()])  # Explanation: assigns an intermediate value used by later computation

                    for a in atom1.GetNeighbors():  # Explanation: iterates over this collection to process each item
                        if a.GetAtomicNum() == a1_max_neigh_num and a.GetIdx() != atom2.GetIdx():  # Explanation: checks this condition to choose the next execution path
                            a1_max_neigh = a.GetIdx()  # Explanation: assigns an intermediate value used by later computation
                    for a in atom2.GetNeighbors():  # Explanation: iterates over this collection to process each item
                        if a.GetAtomicNum() == a2_max_neigh_num and a.GetIdx() != atom1.GetIdx():  # Explanation: checks this condition to choose the next execution path
                            a2_max_neigh = a.GetIdx()  # Explanation: assigns an intermediate value used by later computation

                    if all(a.GetAtomicNum() == a1_max_neigh_num for a in atom1.GetNeighbors() if  # Explanation: checks this condition to choose the next execution path
                           a.GetIdx() != atom2.GetIdx()) and len(  # Explanation: assigns an intermediate value used by later computation
                            [a.GetAtomicNum() for a in atom1.GetNeighbors() if a.GetIdx() != atom2.GetIdx()]) == 2:  # Explanation: assigns an intermediate value used by later computation
                        a11_max_neigh_num = 0  # Explanation: assigns an intermediate value used by later computation
                        for a in atom1.GetNeighbors():  # Explanation: iterates over this collection to process each item
                            if a.GetIdx() != atom2.GetIdx():  # Explanation: checks this condition to choose the next execution path
                                if len([a1.GetAtomicNum() for a1 in a.GetNeighbors() if  # Explanation: checks this condition to choose the next execution path
                                        a1.GetIdx() != atom1.GetIdx()]) == 0:  # Explanation: assigns an intermediate value used by later computation
                                    continue  # Explanation: skips the rest of this loop iteration
                                else:  # Explanation: handles the fallback branch for the preceding condition
                                    a11_max_neigh_num = max(a11_max_neigh_num,  # Explanation: assigns an intermediate value used by later computation
                                                            max([a1.GetAtomicNum() for a1 in a.GetNeighbors() if  # Explanation: executes this statement as part of define executable molecular graph edit actions
                                                                 a1.GetIdx() != atom1.GetIdx()]))  # Explanation: assigns an intermediate value used by later computation
                        for a in atom1.GetNeighbors():  # Explanation: iterates over this collection to process each item
                            if a.GetIdx() != atom2.GetIdx():  # Explanation: checks this condition to choose the next execution path
                                for a1 in a.GetNeighbors():  # Explanation: iterates over this collection to process each item
                                    if a1.GetAtomicNum() == a11_max_neigh_num and a1.GetIdx() != atom1.GetIdx():  # Explanation: checks this condition to choose the next execution path
                                        a1_max_neigh = a.GetIdx()  # Explanation: assigns an intermediate value used by later computation

                    if all(a.GetAtomicNum() == a2_max_neigh_num for a in atom2.GetNeighbors() if  # Explanation: checks this condition to choose the next execution path
                           a.GetIdx() != atom1.GetIdx()) and len(  # Explanation: assigns an intermediate value used by later computation
                            [a.GetAtomicNum() for a in atom2.GetNeighbors() if a.GetIdx() != atom1.GetIdx()]) == 2:  # Explanation: assigns an intermediate value used by later computation
                        a12_max_neigh_num = 0  # Explanation: assigns an intermediate value used by later computation
                        for a in atom2.GetNeighbors():  # Explanation: iterates over this collection to process each item
                            if a.GetIdx() != atom1.GetIdx():  # Explanation: checks this condition to choose the next execution path
                                if len([a2.GetAtomicNum() for a2 in a.GetNeighbors() if  # Explanation: checks this condition to choose the next execution path
                                        a2.GetIdx() != atom2.GetIdx()]) == 0:  # Explanation: assigns an intermediate value used by later computation
                                    continue  # Explanation: skips the rest of this loop iteration
                                else:  # Explanation: handles the fallback branch for the preceding condition
                                    a12_max_neigh_num = max(a12_max_neigh_num,  # Explanation: assigns an intermediate value used by later computation
                                                            max([a2.GetAtomicNum() for a2 in a.GetNeighbors() if  # Explanation: executes this statement as part of define executable molecular graph edit actions
                                                                 a2.GetIdx() != atom2.GetIdx()]))  # Explanation: assigns an intermediate value used by later computation
                        for a in atom2.GetNeighbors():  # Explanation: iterates over this collection to process each item
                            if a.GetIdx() != atom1.GetIdx():  # Explanation: checks this condition to choose the next execution path
                                for a2 in a.GetNeighbors():  # Explanation: iterates over this collection to process each item
                                    if a2.GetAtomicNum() == a12_max_neigh_num and a2.GetIdx() != atom2.GetIdx():  # Explanation: checks this condition to choose the next execution path
                                        a2_max_neigh = a.GetIdx()  # Explanation: assigns an intermediate value used by later computation

                if a1_max_neigh is not None and a2_max_neigh is not None:  # Explanation: checks this condition to choose the next execution path
                    try:  # Explanation: starts a protected block for operations that may fail
                        bond.SetStereoAtoms(a1_max_neigh, a2_max_neigh)  # Explanation: executes this statement as part of define executable molecular graph edit actions
                    except:  # Explanation: handles failures from the preceding try block
                        bond.SetStereoAtoms(a2_max_neigh, a1_max_neigh)  # Explanation: executes this statement as part of define executable molecular graph edit actions

        return mol  # Explanation: returns this computed result to the caller

    def __str__(self):  # Explanation: defines __str__, which define executable molecular graph edit actions
        return 'Terminate'  # Explanation: returns this computed result to the caller
