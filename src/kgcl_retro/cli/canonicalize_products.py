"""
Canonicalize the product SMILES, and then use substructure matching to infer
the correspondence to the original atom-mapped order. This correspondence is then
used to renumber the reactant atoms.
"""
from rdkit import Chem  # Explanation: imports selected names needed to canonicalize reaction SMILES and remap atom numbers
import os  # Explanation: imports os for canonicalize reaction SMILES and remap atom numbers
import argparse  # Explanation: imports argparse for canonicalize reaction SMILES and remap atom numbers
import pandas as pd  # Explanation: imports pandas as pd for canonicalize reaction SMILES and remap atom numbers

from tqdm import tqdm  # Explanation: imports selected names needed to canonicalize reaction SMILES and remap atom numbers
from kgcl_retro.paths import resolve_project_paths  # Explanation: imports shared project-root path resolution for package CLIs.

def canonicalize_prod(p):  # Explanation: defines canonicalize_prod, which canonicalizes a product and assigns atom-map numbers
    import copy  # Explanation: imports copy for canonicalize reaction SMILES and remap atom numbers
    p = copy.deepcopy(p)  # Explanation: assigns an intermediate value used by later computation
    p = canonicalize(p)  # Explanation: assigns an intermediate value used by later computation
    p_mol = Chem.MolFromSmiles(p)  # Explanation: parses a SMILES string into an RDKit molecule

    if p_mol is None:  # Explanation: checks this condition to choose the next execution path
        raise ValueError(f"Invalid SMILES string after canonicalization: {p}")  # Explanation: raises an error when unsupported input is encountered

    for atom in p_mol.GetAtoms():  # Explanation: iterates over this collection to process each item
        atom.SetAtomMapNum(atom.GetIdx() + 1)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
    p = Chem.MolToSmiles(p_mol)  # Explanation: serializes an RDKit molecule back to SMILES
    return p  # Explanation: returns this computed result to the caller


def canonicalize(smiles):  # Explanation: defines canonicalize, which removes atom maps/hydrogens and canonicalizes SMILES
    try:  # Explanation: starts a protected block for operations that may fail
        tmp = Chem.MolFromSmiles(smiles)  # Explanation: parses a SMILES string into an RDKit molecule
    except:  # Explanation: handles failures from the preceding try block
        print('no mol', flush=True)  # Explanation: prints progress or diagnostic information
        return smiles  # Explanation: returns this computed result to the caller
    if tmp is None:  # Explanation: checks this condition to choose the next execution path
        return smiles  # Explanation: returns this computed result to the caller
    tmp = Chem.RemoveHs(tmp)  # Explanation: removes explicit hydrogen atoms before canonicalization
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]  # Explanation: executes this list-comprehension side effect over molecule atoms
    return Chem.MolToSmiles(tmp)  # Explanation: returns this computed result to the caller


def fix_charge(mol):  # Explanation: defines fix_charge, which fixes common charge and hydrogen states
    for atom in mol.GetAtoms():  # Explanation: iterates over this collection to process each item
        explicit_hs = atom.GetNumExplicitHs()  # Explanation: assigns an intermediate value used by later computation
        charge = atom.GetFormalCharge()  # Explanation: assigns an intermediate value used by later computation
        bond_vals = int(sum([b.GetBondTypeAsDouble()  # Explanation: computes an intermediate value for molecular graph editing
                        for b in atom.GetBonds()]))  # Explanation: iterates over this collection to process each item
        if atom.GetSymbol() == 'O' and bond_vals == 1 and charge == -1 and explicit_hs == 0:  # Explanation: checks this condition to choose the next execution path
            if atom.GetNeighbors()[0].GetSymbol() != 'N':  # Explanation: checks this condition to choose the next execution path
                atom.SetFormalCharge(0)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
                atom.SetNumExplicitHs(1)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers

        if atom.GetSymbol() == 'N' and bond_vals == 1 and charge == 1 and explicit_hs == 3:  # Explanation: checks this condition to choose the next execution path
            atom.SetFormalCharge(0)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
            atom.SetNumExplicitHs(2)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers

        if atom.GetSymbol() == 'N' and bond_vals == 0 and charge == 1 and explicit_hs == 4:  # Explanation: checks this condition to choose the next execution path
            atom.SetFormalCharge(0)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
            atom.SetNumExplicitHs(3)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers

        if atom.GetSymbol() == 'N' and bond_vals == 2 and charge == 1 and explicit_hs == 2:  # Explanation: checks this condition to choose the next execution path
            atom.SetFormalCharge(0)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
            atom.SetNumExplicitHs(1)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers

        if atom.GetSymbol() == 'S' and charge == -1 and explicit_hs == 0 and bond_vals == 1:  # Explanation: checks this condition to choose the next execution path
            atom.SetNumExplicitHs(1)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
            atom.SetFormalCharge(0)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
    return mol  # Explanation: returns this computed result to the caller


def infer_correspondence(p):  # Explanation: defines infer_correspondence, which matches original atom maps to canonical product atom maps
    orig_mol = Chem.MolFromSmiles(p)  # Explanation: parses a SMILES string into an RDKit molecule
    canon_mol = Chem.MolFromSmiles(canonicalize_prod(p))  # Explanation: parses a SMILES string into an RDKit molecule

    matches = list(canon_mol.GetSubstructMatches(orig_mol,maxMatches=1))  # 若不设置，高度对称的大分子搜索将超时  # Explanation: assigns an intermediate value used by later computation
    idx_amap = {atom.GetIdx(): atom.GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
                for atom in orig_mol.GetAtoms()}  # Explanation: iterates over this collection to process each item

    correspondence = {}  # Explanation: assigns an intermediate value used by later computation
    if matches:  # Explanation: checks this condition to choose the next execution path
        for idx, match_idx in enumerate(matches[0]):  # Explanation: iterates over this collection to process each item
            match_anum = canon_mol.GetAtomWithIdx(match_idx).GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
            old_anum = idx_amap[idx]  # Explanation: assigns an intermediate value used by later computation
            correspondence[old_anum] = match_anum  # Explanation: assigns an intermediate value used by later computation
    else:  # Explanation: handles the fallback branch for the preceding condition
        print(f"No matches found for SMILES: {p}")  # Explanation: prints progress or diagnostic information
    return correspondence  # Explanation: returns this computed result to the caller


def remap_rxn_smi(rxn_smi):  # Explanation: defines remap_rxn_smi, which remaps a full reaction SMILES consistently

    r, p = rxn_smi.split(">>")  # Explanation: assigns an intermediate value used by later computation
    canon_mol = Chem.MolFromSmiles(canonicalize_prod(p))  # Explanation: parses a SMILES string into an RDKit molecule
    correspondence = infer_correspondence(p)  # Explanation: assigns an intermediate value used by later computation
    rmol = Chem.MolFromSmiles(r)  # Explanation: parses a SMILES string into an RDKit molecule
    if rmol is None or rmol.GetNumAtoms() <= 1:  # Explanation: checks this condition to choose the next execution path
        return rxn_smi, None  # Explanation: returns this computed result to the caller

    for atom in rmol.GetAtoms():  # Explanation: iterates over this collection to process each item
        atomnum = atom.GetAtomMapNum()  # Explanation: assigns an intermediate value used by later computation
        if atomnum in correspondence:  # Explanation: checks this condition to choose the next execution path
            newatomnum = correspondence[atomnum]  # Explanation: assigns an intermediate value used by later computation
            atom.SetAtomMapNum(newatomnum)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers

    max_amap = max([atom.GetAtomMapNum() for atom in rmol.GetAtoms()])  # Explanation: assigns an intermediate value used by later computation
    for atom in rmol.GetAtoms():  # Explanation: iterates over this collection to process each item
        if atom.GetAtomMapNum() == 0:  # Explanation: checks this condition to choose the next execution path
            atom.SetAtomMapNum(max_amap + 1)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
            max_amap += 1  # Explanation: assigns an intermediate value used by later computation

    rmol = fix_charge(rmol)  # Explanation: assigns an intermediate value used by later computation
    canon_mol = fix_charge(canon_mol)  # Explanation: assigns an intermediate value used by later computation

    rmol = Chem.MolFromSmiles(Chem.MolToSmiles(rmol))  # Explanation: parses a SMILES string into an RDKit molecule
    rxn_smi_new = Chem.MolToSmiles(rmol) + ">>" + Chem.MolToSmiles(canon_mol)  # Explanation: serializes an RDKit molecule back to SMILES
    return rxn_smi_new, correspondence  # Explanation: returns this computed result to the caller


def main():  # Explanation: defines main, which runs this script from command-line arguments
    parser = argparse.ArgumentParser()  # Explanation: creates command-line argument parser
    parser.add_argument('--dataset', type=str, default='USPTO_full',  # Explanation: chooses which USPTO dataset split to use
                        help='dataset: USPTO_50k or USPTO_full')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--mode', type=str, default='test',  # Explanation: selects train, valid, or test split
                        help='Type of dataset being prepared: train or valid or test')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--root_dir', type=str, default='.',  # Explanation: selects the root directory containing data and experiments.
                        help='Repository/data root containing data/ and experiments/')  # Explanation: documents the package-relative root directory option.
    args = parser.parse_args()  # Explanation: parses command-line options

    args.dataset = args.dataset.lower()  # Explanation: assigns an intermediate value used by later computation
    paths = resolve_project_paths(args.root_dir)  # Explanation: resolves the root directory used by package CLI file operations.
    datadir = str(paths.dataset_dir(args.dataset))  # Explanation: builds the selected dataset directory from the resolved root.
    new_file = f'canonicalized_{args.mode}.csv'  # Explanation: assigns an intermediate value used by later computation
    filename = f'raw_{args.mode}.csv'  # Explanation: assigns an intermediate value used by later computation
    df = pd.read_csv(os.path.join(datadir, filename))  # Explanation: builds a filesystem path
    print(f"Processing file of size: {len(df)}")  # Explanation: prints progress or diagnostic information

    if args.dataset == 'uspto_50k':  # Explanation: checks this condition to choose the next execution path
        new_dict = {'id': [], 'class': [], 'reactants>reagents>production': []}  # Explanation: assigns an intermediate value used by later computation
    else:  # Explanation: handles the fallback branch for the preceding condition
        new_dict = {'id': [], 'reactants>reagents>production': []}  # Explanation: assigns an intermediate value used by later computation

    for idx in tqdm(range(len(df)), desc="Processing reactions"):  # Explanation: iterates over this collection to process each item
        element = df.loc[idx]  # Explanation: assigns an intermediate value used by later computation
        if args.dataset == 'uspto_50k':  # Explanation: checks this condition to choose the next execution path
            uspto_id, class_id, rxn_smi = element['id'], element['class'], element['reactants>reagents>production']  # Explanation: assigns an intermediate value used by later computation
        else:  # Explanation: handles the fallback branch for the preceding condition
            uspto_id, rxn_smi = element['id'], element['reactants>reagents>production']  # Explanation: assigns an intermediate value used by later computation

        rxn_smi_new, _ = remap_rxn_smi(rxn_smi)  # Explanation: computes an intermediate value for molecular graph editing
        new_dict['id'].append(uspto_id)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
        if args.dataset == 'uspto_50k':  # Explanation: checks this condition to choose the next execution path
            new_dict['class'].append(class_id)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
        new_dict['reactants>reagents>production'].append(rxn_smi_new)  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers

    new_df = pd.DataFrame.from_dict(new_dict)  # Explanation: assigns an intermediate value used by later computation
    new_df.to_csv(os.path.join(datadir, new_file), index=False)  # Explanation: builds a filesystem path


if __name__ == "__main__":  # Explanation: runs the CLI entry point only when this file is executed directly
    main()  # Explanation: executes this statement as part of canonicalize reaction SMILES and remap atom numbers
