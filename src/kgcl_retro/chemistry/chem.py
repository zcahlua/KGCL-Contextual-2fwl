from typing import Dict, Tuple  # Explanation: imports selected names needed to provide RDKit chemistry helper operations
from rdkit import Chem  # Explanation: imports selected names needed to provide RDKit chemistry helper operations
from rdkit.Chem import Mol, RWMol, rdchem  # Explanation: imports selected names needed to provide RDKit chemistry helper operations

MAX_BONDS = {'C': 4, 'N': 3, 'O': 2, 'Br': 1,  # Explanation: defines a module-level constant used by the pipeline
             'Cl': 1, 'F': 1, 'I': 1, 'Li': 1, 'Na': 1, 'K': 1}  # Explanation: executes this statement as part of provide RDKit chemistry helper operations


def get_atom_info(mol: Mol) -> Dict:  # Explanation: defines get_atom_info, which extracts atom hydrogen and chirality information
    if mol is None:  # Explanation: checks this condition to choose the next execution path
        return {}  # Explanation: returns this computed result to the caller

    atom_info = {}  # Explanation: computes an intermediate value for molecular graph editing
    for atom in mol.GetAtoms():  # Explanation: iterates over this collection to process each item
        feat = [atom.GetNumExplicitHs(), int(atom.GetChiralTag())]  # Explanation: assigns an intermediate value used by later computation
        amap_num = atom.GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
        atom_info[amap_num] = tuple(feat)  # Explanation: computes an intermediate value for molecular graph editing
    return atom_info  # Explanation: returns this computed result to the caller


def get_atom_Chiral(mol: Mol) -> Dict:  # Explanation: defines get_atom_Chiral, which extracts atom chirality tags
    if mol is None:  # Explanation: checks this condition to choose the next execution path
        return {}  # Explanation: returns this computed result to the caller

    atom_Chiral = {}  # Explanation: computes an intermediate value for molecular graph editing
    for atom in mol.GetAtoms():  # Explanation: iterates over this collection to process each item
        amap_num = atom.GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
        atom_Chiral[amap_num] = atom.GetChiralTag()  # Explanation: computes an intermediate value for molecular graph editing
    return atom_Chiral  # Explanation: returns this computed result to the caller


def get_bond_info(mol: Mol) -> Dict:  # Explanation: defines get_bond_info, which extracts bond type and stereo information
    if mol is None:  # Explanation: checks this condition to choose the next execution path
        return {}  # Explanation: returns this computed result to the caller

    bond_info = {}  # Explanation: computes an intermediate value for molecular graph editing
    for bond in mol.GetBonds():  # Explanation: iterates over this collection to process each item
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
        bt = int(bond.GetBondType())  # Explanation: assigns an intermediate value used by later computation
        st = int(bond.GetStereo())  # Explanation: assigns an intermediate value used by later computation
        bond_atoms = sorted([a1, a2])  # Explanation: computes an intermediate value for molecular graph editing
        bond_info[tuple(bond_atoms)] = [bt, st]  # Explanation: computes an intermediate value for molecular graph editing
    return bond_info  # Explanation: returns this computed result to the caller


def get_bond_stereo(mol: Mol) -> Dict:  # Explanation: defines get_bond_stereo, which extracts bond stereo states
    if mol is None:  # Explanation: checks this condition to choose the next execution path
        return {}  # Explanation: returns this computed result to the caller

    bond_stereo = {}  # Explanation: computes an intermediate value for molecular graph editing
    for bond in mol.GetBonds():  # Explanation: iterates over this collection to process each item
        a1, a2 = bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
        bond_atoms = sorted([a1, a2])  # Explanation: computes an intermediate value for molecular graph editing
        bond_stereo[tuple(bond_atoms)] = bond.GetStereo()  # Explanation: computes an intermediate value for molecular graph editing
    return bond_stereo  # Explanation: returns this computed result to the caller


def align_kekulize_pairs(r_mol: Mol, p_mol: Mol) -> Tuple[Mol, Mol]:  # Explanation: defines align_kekulize_pairs, which aligns kekulized reactant/product bond forms
    prod_old = get_bond_info(p_mol)  # Explanation: computes an intermediate value for molecular graph editing
    Chem.Kekulize(p_mol)  # Explanation: converts aromatic bonds into kekulized form
    prod_new = get_bond_info(p_mol)  # Explanation: computes an intermediate value for molecular graph editing

    react_old = get_bond_info(r_mol)  # Explanation: computes an intermediate value for molecular graph editing
    Chem.Kekulize(r_mol)  # Explanation: converts aromatic bonds into kekulized form
    react_new = get_bond_info(r_mol)  # Explanation: computes an intermediate value for molecular graph editing

    r_mol = Chem.RWMol(r_mol)  # Explanation: creates an editable RDKit molecule
    r_amap_idx = {atom.GetAtomMapNum(): atom.GetIdx()  # Explanation: assigns an intermediate value used by later computation
                  for atom in r_mol.GetAtoms()}  # Explanation: iterates over this collection to process each item
    for bond in prod_new:  # Explanation: iterates over this collection to process each item
        if bond in react_new and (react_old[bond][0] == prod_old[bond][0]) and (react_new[bond][0] != prod_new[bond][0]):  # Explanation: checks this condition to choose the next execution path
            idx1, idx2 = r_amap_idx[bond[0]], r_amap_idx[bond[1]]  # Explanation: assigns an intermediate value used by later computation
            bt = prod_new[bond][0]  # Explanation: assigns an intermediate value used by later computation
            b_type = rdchem.BondType.values[bt]  # Explanation: assigns an intermediate value used by later computation
            r_mol.GetBondBetweenAtoms(idx1, idx2).SetBondType(b_type)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

    return r_mol.GetMol(), p_mol  # Explanation: returns this computed result to the caller


def get_atom_idx(mol: RWMol or Mol, atom_map: int) -> int:  # Explanation: defines get_atom_idx, which finds an atom by atom-map number
    for i, a in enumerate(mol.GetAtoms()):  # Explanation: iterates over this collection to process each item
        if a.GetAtomMapNum() == atom_map:  # Explanation: checks this condition to choose the next execution path
            return i  # Explanation: returns this computed result to the caller
    raise ValueError(f'No atom with map number: {atom_map}')  # Explanation: raises an error when unsupported input is encountered


def attach_lg(main_mol: Mol, lg_mol: Mol, attach_atom_map: int) -> Mol:  # Explanation: defines attach_lg, which attaches a leaving group to a molecule
    combined_mol = Chem.CombineMols(main_mol, lg_mol)  # Explanation: combines molecule fragments before attachment
    rw_mol = Chem.RWMol(Chem.Mol(combined_mol))  # Explanation: creates an editable RDKit molecule

    lg_attach_num = 0  # Explanation: assigns an intermediate value used by later computation
    for atom in rw_mol.GetAtoms():  # Explanation: iterates over this collection to process each item
        if atom.GetSymbol() == '*':  # Explanation: checks this condition to choose the next execution path
            lg_attach_num += 1  # Explanation: assigns an intermediate value used by later computation

    if lg_attach_num == 1:  # Explanation: checks this condition to choose the next execution path
        for atom in rw_mol.GetAtoms():  # Explanation: iterates over this collection to process each item
            if atom.GetSymbol() == '*':  # Explanation: checks this condition to choose the next execution path
                remove_idx = atom.GetIdx()  # Explanation: assigns an intermediate value used by later computation
                lg_attach_atom = atom.GetNeighbors()  # Explanation: assigns an intermediate value used by later computation
                lg_attach_atom[0].SetAtomMapNum(500)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                bond = atom.GetBonds()  # Explanation: assigns an intermediate value used by later computation
                bt = bond[0].GetBondType()  # Explanation: assigns an intermediate value used by later computation

        rw_mol.RemoveAtom(remove_idx)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()  # Explanation: assigns an intermediate value used by later computation
                    if atom.GetAtomMapNum() != 0}  # Explanation: checks this condition to choose the next execution path

        attach_atom_idx = amap_idx[attach_atom_map]  # Explanation: assigns an intermediate value used by later computation
        lg_attach_idx = amap_idx[500]  # Explanation: assigns an intermediate value used by later computation

        rw_mol.AddBond(attach_atom_idx, lg_attach_idx, bt)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
        rw_mol.GetAtomWithIdx(amap_idx[500]).ClearProp('molAtomMapNumber')  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

    else:  # Explanation: handles the fallback branch for the preceding condition
        lg_attach_amap = 500  # Explanation: assigns an intermediate value used by later computation
        remove_atommap = 1000  # Explanation: assigns an intermediate value used by later computation
        for atom in rw_mol.GetAtoms():  # Explanation: iterates over this collection to process each item
            if atom.GetSymbol() == '*':  # Explanation: checks this condition to choose the next execution path
                atom.SetAtomMapNum(remove_atommap)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                lg_attach_atom = atom.GetNeighbors()  # Explanation: assigns an intermediate value used by later computation
                lg_attach_atom[0].SetAtomMapNum(lg_attach_amap)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                lg_attach_amap += 1  # Explanation: assigns an intermediate value used by later computation
                remove_atommap += 1  # Explanation: assigns an intermediate value used by later computation

        amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()  # Explanation: assigns an intermediate value used by later computation
                    if atom.GetAtomMapNum() != 0}  # Explanation: checks this condition to choose the next execution path

        for num in range(lg_attach_num):  # Explanation: iterates over this collection to process each item
            lg_attach_amap = 500  # Explanation: assigns an intermediate value used by later computation
            remove_atommap = 1000  # Explanation: assigns an intermediate value used by later computation
            remove_idx = amap_idx[remove_atommap + num]  # Explanation: assigns an intermediate value used by later computation
            remove_atom = rw_mol.GetAtomWithIdx(remove_idx)  # Explanation: assigns an intermediate value used by later computation
            bond = remove_atom.GetBonds()  # Explanation: assigns an intermediate value used by later computation
            bt = bond[0].GetBondType()  # Explanation: assigns an intermediate value used by later computation

            rw_mol.RemoveAtom(remove_idx)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
            amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()  # Explanation: assigns an intermediate value used by later computation
                        if atom.GetAtomMapNum() != 0}  # Explanation: checks this condition to choose the next execution path

            attach_atom_idx = amap_idx[attach_atom_map]  # Explanation: assigns an intermediate value used by later computation
            lg_attach_idx = amap_idx[lg_attach_amap + num]  # Explanation: assigns an intermediate value used by later computation

            rw_mol.AddBond(attach_atom_idx, lg_attach_idx, bt)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
            rw_mol.GetAtomWithIdx(  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                amap_idx[lg_attach_amap + num]).ClearProp('molAtomMapNumber')  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

    max_amap = max([atom.GetAtomMapNum() for atom in rw_mol.GetAtoms()])  # Explanation: assigns an intermediate value used by later computation
    for atom in rw_mol.GetAtoms():  # Explanation: iterates over this collection to process each item
        if atom.GetAtomMapNum() == 0:  # Explanation: checks this condition to choose the next execution path
            atom.SetAtomMapNum(max_amap + 1)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
            atom.SetNoImplicit(True)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
            max_amap += 1  # Explanation: assigns an intermediate value used by later computation

    new_mol = rw_mol.GetMol()  # Explanation: assigns an intermediate value used by later computation

    return new_mol  # Explanation: returns this computed result to the caller


def fix_Hs_Charge(mol: Mol) -> Mol:  # Explanation: defines fix_Hs_Charge, which repairs hydrogen and charge states after edits
    # fix explicit Hs and charge
    for atom in mol.GetAtoms():  # Explanation: iterates over this collection to process each item
        atom_symbol = atom.GetSymbol()  # Explanation: computes an intermediate value for molecular graph editing
        explicit_hs = atom.GetNumExplicitHs()  # Explanation: assigns an intermediate value used by later computation
        charge = atom.GetFormalCharge()  # Explanation: assigns an intermediate value used by later computation
        bond_vals = int(sum([b.GetBondTypeAsDouble()  # Explanation: computes an intermediate value for molecular graph editing
                        for b in atom.GetBonds()]))  # Explanation: iterates over this collection to process each item

        if not atom.IsInRing():  # Explanation: checks this condition to choose the next execution path
            atom.SetIsAromatic(False)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

        if charge == 0:  # Explanation: checks this condition to choose the next execution path
            if atom_symbol in MAX_BONDS and explicit_hs + bond_vals > MAX_BONDS[atom_symbol]:  # Explanation: checks this condition to choose the next execution path
                num = int(explicit_hs + bond_vals - MAX_BONDS[atom_symbol])  # Explanation: assigns an intermediate value used by later computation
                for i in range(num):  # Explanation: iterates over this collection to process each item
                    if explicit_hs > 0:  # Explanation: checks this condition to choose the next execution path
                        explicit_hs -= 1  # Explanation: assigns an intermediate value used by later computation
                        atom.SetNumExplicitHs(explicit_hs)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                    else:  # Explanation: handles the fallback branch for the preceding condition
                        atom.SetFormalCharge(1)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

            elif atom_symbol in MAX_BONDS and explicit_hs + bond_vals < MAX_BONDS[atom_symbol]:  # Explanation: checks an alternate condition after the previous branch failed
                num = int(MAX_BONDS[atom_symbol] - explicit_hs - bond_vals)  # Explanation: assigns an intermediate value used by later computation
                for i in range(num):  # Explanation: iterates over this collection to process each item
                    explicit_hs += 1  # Explanation: assigns an intermediate value used by later computation
                    atom.SetNumExplicitHs(explicit_hs)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

            # "-N=N+=N-"
            if atom_symbol == 'N' and len([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) == 1 and bond_vals == 2 and atom.GetNeighbors()[0].GetSymbol() == 'N':  # Explanation: checks this condition to choose the next execution path
                atom.SetNumExplicitHs(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                atom.SetFormalCharge(-1)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

            # "NC-"
            if atom_symbol == 'C' and len([b.GetBondTypeAsDouble() for b in atom.GetBonds()]) == 1 and bond_vals == 3 and atom.GetNeighbors()[0].GetSymbol() == 'N':  # Explanation: checks this condition to choose the next execution path
                atom.SetNumExplicitHs(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                atom.SetFormalCharge(-1)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

            if atom_symbol == 'S' and explicit_hs == 0 and bond_vals == 1:  # Explanation: checks this condition to choose the next execution path
                atom.SetNumExplicitHs(1)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

            if atom_symbol == 'S' and explicit_hs == 1 and bond_vals in [2, 4, 6]:  # Explanation: checks this condition to choose the next execution path
                atom.SetNumExplicitHs(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

            if atom_symbol == 'P':  # 'P(OCC)3'  # Explanation: checks this condition to choose the next execution path
                bond_vals = [bond.GetBondTypeAsDouble()  # Explanation: computes an intermediate value for molecular graph editing
                             for bond in atom.GetBonds()]  # Explanation: iterates over this collection to process each item
                if sum(bond_vals) == 3 and len(bond_vals) == 3:  # Explanation: checks this condition to choose the next execution path
                    atom.SetNumExplicitHs(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                if sum(bond_vals) == 4 and len(bond_vals) == 4:  # Explanation: checks this condition to choose the next execution path
                    atom.SetFormalCharge(1)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

            if atom_symbol == 'Sn':  # Explanation: checks this condition to choose the next execution path
                if explicit_hs == 0 and bond_vals == 3:  # Explanation: checks this condition to choose the next execution path
                    atom.SetNumExplicitHs(1)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                if explicit_hs == 1 and bond_vals == 4:  # Explanation: checks this condition to choose the next execution path
                    atom.SetNumExplicitHs(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

        else:  # Explanation: handles the fallback branch for the preceding condition
            if atom_symbol in MAX_BONDS and explicit_hs + bond_vals == MAX_BONDS[atom_symbol]:  # Explanation: checks this condition to choose the next execution path
                atom.SetFormalCharge(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

            if atom_symbol == 'O':  # Explanation: checks this condition to choose the next execution path
                bond_vals = bond_vals + explicit_hs  # Explanation: computes an intermediate value for molecular graph editing
                if bond_vals == 1 and charge == -1 and atom.GetNeighbors()[0].GetSymbol() != 'N':  # Explanation: checks this condition to choose the next execution path
                    atom.SetFormalCharge(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                    atom.SetNumExplicitHs(1)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

            if atom_symbol == 'N':  # Explanation: checks this condition to choose the next execution path
                if bond_vals == 4 and explicit_hs == 0 and charge == -1:  # Explanation: checks this condition to choose the next execution path
                    atom.SetFormalCharge(1)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                if bond_vals == 3 and explicit_hs == 1 and charge == -1:  # Explanation: checks this condition to choose the next execution path
                    atom.SetFormalCharge(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                    atom.SetNumExplicitHs(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                if bond_vals == 3 and explicit_hs == 2 and charge == 1:  # Explanation: checks this condition to choose the next execution path
                    atom.SetFormalCharge(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations
                    atom.SetNumExplicitHs(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

    for atom in mol.GetAtoms():  # Dealing with the problem 'C+'  # Explanation: iterates over this collection to process each item
        if atom.GetSymbol() == 'C' and atom.GetFormalCharge() == 1:  # Explanation: checks this condition to choose the next execution path
            atom.SetFormalCharge(0)  # Explanation: executes this statement as part of provide RDKit chemistry helper operations

    return mol  # Explanation: returns this computed result to the caller
