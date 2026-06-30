import argparse  # Explanation: imports argparse for extract reaction edit labels and vocabularies
import os  # Explanation: imports os for extract reaction edit labels and vocabularies
import sys  # Explanation: imports sys for extract reaction edit labels and vocabularies
from collections import Counter  # Explanation: imports selected names needed to extract reaction edit labels and vocabularies
from typing import Any, List  # Explanation: imports selected names needed to extract reaction edit labels and vocabularies

import joblib  # Explanation: imports joblib for extract reaction edit labels and vocabularies
import pandas as pd  # Explanation: imports pandas as pd for extract reaction edit labels and vocabularies
from rdkit import Chem  # Explanation: imports selected names needed to extract reaction edit labels and vocabularies

from kgcl_retro.chemistry.edits import generate_reaction_edits  # Explanation: imports packaged edit extraction for preprocessing reactions.
from kgcl_retro.paths import resolve_project_paths  # Explanation: imports shared project-root path resolution for package CLIs.


def check_edits(edits: List):  # Explanation: defines check_edits, which filters unsupported edit types
    for edit in edits:  # Explanation: iterates over this collection to process each item
        if edit[0] == 'Add Bond':  # Explanation: checks this condition to choose the next execution path
            return False  # Explanation: returns this computed result to the caller

    return True  # Explanation: returns this computed result to the caller


def preprocessing(rxns: List, args: Any, rxn_classes: List = [], rxns_id=[]) -> None:  # Explanation: defines preprocessing, which builds ReactionData and edit vocabularies
    """
    preprocess reactions data to get edits
    """
    rxns_data = []  # Explanation: assigns an intermediate value used by later computation
    counter = []  # Explanation: assigns an intermediate value used by later computation
    all_edits = {}  # Explanation: assigns an intermediate value used by later computation

    paths = resolve_project_paths(getattr(args, 'root_dir', '.'))  # Explanation: resolves the root directory used by package CLI file operations.
    savedir = os.path.join(str(paths.dataset_dir(args.dataset)), args.mode)  # Explanation: builds the processed split output directory from the resolved root.
    os.makedirs(savedir, exist_ok=True)  # Explanation: creates output directories when needed

    for idx, rxn_smi in enumerate(rxns):  # Explanation: iterates over this collection to process each item
        r, p = rxn_smi.split('>>')  # Explanation: assigns an intermediate value used by later computation
        prod_mol = Chem.MolFromSmiles(p)  # Explanation: parses a SMILES string into an RDKit molecule

        if (prod_mol is None) or (prod_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):  # Explanation: checks this condition to choose the next execution path
            print(  # Explanation: prints progress or diagnostic information
                f'Product has 0 or 1 atom or 1 bond, Skipping reaction {idx}')  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
            print()  # Explanation: prints progress or diagnostic information
            sys.stdout.flush()  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
            continue  # Explanation: skips the rest of this loop iteration

        react_mol = Chem.MolFromSmiles(r)  # Explanation: parses a SMILES string into an RDKit molecule

        if (react_mol is None) or (react_mol.GetNumAtoms() <= 1) or (prod_mol.GetNumBonds() <= 1):  # Explanation: checks this condition to choose the next execution path
            print(  # Explanation: prints progress or diagnostic information
                f'Reactant has 0 or 1 atom or 1 bond, Skipping reaction {idx}')  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
            print()  # Explanation: prints progress or diagnostic information
            sys.stdout.flush()  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
            continue  # Explanation: skips the rest of this loop iteration

        try:  # Explanation: starts a protected block for operations that may fail
            if args.dataset == 'uspto_50k':  # Explanation: checks this condition to choose the next execution path
                rxn_data = generate_reaction_edits(rxn_smi, kekulize=args.kekulize, rxn_class=int(  # Explanation: computes an intermediate value for molecular graph editing
                    rxn_classes[idx]) - 1, rxn_id=rxns_id[idx])  # Explanation: computes an intermediate value for molecular graph editing
            else:  # Explanation: handles the fallback branch for the preceding condition
                rxn_data = generate_reaction_edits(  # Explanation: computes an intermediate value for molecular graph editing
                    rxn_smi, kekulize=args.kekulize)  # Explanation: computes an intermediate value for molecular graph editing
        except:  # Explanation: handles failures from the preceding try block
            print(f'Failed to extract reaction data, skipping reaction {idx}')  # Explanation: prints progress or diagnostic information
            print()  # Explanation: prints progress or diagnostic information
            sys.stdout.flush()  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
            continue  # Explanation: skips the rest of this loop iteration

        edits_accepted = check_edits(rxn_data.edits)  # Explanation: assigns an intermediate value used by later computation
        if not edits_accepted:  # Explanation: checks this condition to choose the next execution path
            print(f'Edit: Add new bond. Skipping reaction {idx}')  # Explanation: prints progress or diagnostic information
            print()  # Explanation: prints progress or diagnostic information
            sys.stdout.flush()  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
            continue  # Explanation: skips the rest of this loop iteration

        if args.dataset == 'uspto_full':  # Explanation: checks this condition to choose the next execution path
            if len(rxn_data.edits) > 9 or len(rxn_data.edits) == 1:  # Explanation: checks this condition to choose the next execution path
                print(f'Edits step exceed max_steps or edit step is 1. Skipping reaction {idx}')  # Explanation: prints progress or diagnostic information
                print()  # Explanation: prints progress or diagnostic information
                sys.stdout.flush()  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                continue  # Explanation: skips the rest of this loop iteration

        if args.dataset == 'uspto_mit':  # Explanation: checks this condition to choose the next execution path
            if len(rxn_data.edits) > 9 or len(rxn_data.edits) == 1:  # Explanation: checks this condition to choose the next execution path
                print(f'Edits step exceed max_steps or edit step is 1. Skipping reaction {idx}')  # Explanation: prints progress or diagnostic information
                print()  # Explanation: prints progress or diagnostic information
                sys.stdout.flush()  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                continue  # Explanation: skips the rest of this loop iteration

        rxns_data.append(rxn_data)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies

        if (idx % args.print_every == 0) and idx:  # Explanation: checks this condition to choose the next execution path
            print(f'{idx}/{len(rxns)} {args.mode} reactions processed.')  # Explanation: prints progress or diagnostic information
            sys.stdout.flush()  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies

    print(f'All {args.mode} reactions complete.')  # Explanation: prints progress or diagnostic information
    sys.stdout.flush()  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies

    save_file = os.path.join(savedir, f'{args.mode}.file')  # Explanation: builds a filesystem path
    if args.kekulize:  # Explanation: checks this condition to choose the next execution path
        save_file += '.kekulized'  # Explanation: assigns an intermediate value used by later computation

    if args.mode == 'train':  # Explanation: checks this condition to choose the next execution path
        for idx, rxn_data in enumerate(rxns_data):  # Explanation: iterates over this collection to process each item
            for edit in rxn_data.edits:  # Explanation: iterates over this collection to process each item
                if edit not in all_edits:  # Explanation: checks this condition to choose the next execution path
                    all_edits[edit] = 1  # Explanation: assigns an intermediate value used by later computation
                else:  # Explanation: handles the fallback branch for the preceding condition
                    all_edits[edit] += 1  # Explanation: assigns an intermediate value used by later computation

        atom_edits = []  # Explanation: computes an intermediate value for molecular graph editing
        bond_edits = []  # Explanation: computes an intermediate value for molecular graph editing
        lg_edits = []  # Explanation: assigns an intermediate value used by later computation
        atom_lg_edits = []  # Explanation: computes an intermediate value for molecular graph editing

        if args.dataset == 'uspto_50k':  # Explanation: checks this condition to choose the next execution path
            for edit, num in all_edits.items():  # Explanation: iterates over this collection to process each item
                if edit[0] == 'Change Atom':  # Explanation: checks this condition to choose the next execution path
                    atom_edits.append(edit)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                    atom_lg_edits.append(edit)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':  # Explanation: checks an alternate condition after the previous branch failed
                    bond_edits.append(edit)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                elif edit[0] == 'Attaching LG':  # Explanation: checks an alternate condition after the previous branch failed
                    lg_edits.append(edit)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
            atom_lg_edits.extend(lg_edits)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies

        elif args.dataset == 'uspto_full':  # Explanation: checks an alternate condition after the previous branch failed
            for edit, num in all_edits.items():  # Explanation: iterates over this collection to process each item
                if edit[0] == 'Change Atom':  # Explanation: checks this condition to choose the next execution path
                    atom_edits.append(edit)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                    atom_lg_edits.append(edit)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':  # Explanation: checks an alternate condition after the previous branch failed
                    bond_edits.append(edit)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                elif edit[0] == 'Attaching LG' and num >= 50:  # Explanation: checks an alternate condition after the previous branch failed
                    lg_edits.append(edit)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
            atom_lg_edits.extend(lg_edits)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies

        # elif args.dataset == 'uspto_mit':
        #     for edit, num in all_edits.items():
        #         if edit[0] == 'Change Atom':
        #             atom_edits.append(edit)
        #             atom_lg_edits.append(edit)
        #         elif edit[0] == 'Delete Bond' or edit[0] == 'Change Bond' or edit[0] == 'Add Bond':
        #             bond_edits.append(edit)
        #         elif edit[0] == 'Attaching LG' and num >= 20:
        #             lg_edits.append(edit)
        #     atom_lg_edits.extend(lg_edits)

        print(atom_edits)  # Explanation: prints progress or diagnostic information
        print(bond_edits)  # Explanation: prints progress or diagnostic information
        print(lg_edits)  # Explanation: prints progress or diagnostic information

        filter_rxns_data = []  # Explanation: assigns an intermediate value used by later computation
        for idx, rxn_data in enumerate(rxns_data):  # Explanation: iterates over this collection to process each item
            for edit in rxn_data.edits:  # Explanation: iterates over this collection to process each item
                if edit[0] == 'Attaching LG' and edit not in lg_edits:  # Explanation: checks this condition to choose the next execution path
                    print(  # Explanation: prints progress or diagnostic information
                        f'The number of {edit} in training set is very small, skipping reaction')  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                    rxn_data = None  # Explanation: computes an intermediate value for molecular graph editing
            if rxn_data is not None:  # Explanation: checks this condition to choose the next execution path
                counter.append(len(rxn_data.edits))  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
                filter_rxns_data.append(rxn_data)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies

        print(Counter(counter))  # Explanation: prints progress or diagnostic information

        joblib.dump(filter_rxns_data, save_file, compress=3)  # Explanation: saves processed dataset or vocabulary objects
        joblib.dump(atom_edits, os.path.join(savedir, 'atom_vocab.txt'))  # Explanation: saves processed dataset or vocabulary objects
        joblib.dump(bond_edits, os.path.join(savedir, 'bond_vocab.txt'))  # Explanation: saves processed dataset or vocabulary objects
        joblib.dump(lg_edits, os.path.join(savedir, 'lg_vocab.txt'))  # Explanation: saves processed dataset or vocabulary objects
        joblib.dump(atom_lg_edits, os.path.join(savedir, 'atom_lg_vocab.txt'))  # Explanation: saves processed dataset or vocabulary objects
    else:  # Explanation: handles the fallback branch for the preceding condition
        train_dir = os.path.join(str(paths.dataset_dir(args.dataset)), 'train')  # Explanation: locates the training split vocabulary directory from the resolved root.
        bond_vocab_file = os.path.join(train_dir, 'bond_vocab.txt')  # Explanation: builds the bond vocabulary path from the training split directory.
        atom_vocab_file = os.path.join(train_dir, 'atom_lg_vocab.txt')  # Explanation: builds the atom and leaving-group vocabulary path from the training split directory.
        bond_vocab = joblib.load(bond_vocab_file)  # Explanation: loads processed dataset or vocabulary objects
        atom_vocab = joblib.load(atom_vocab_file)  # Explanation: loads processed dataset or vocabulary objects
        bond_vocab.extend(atom_vocab)  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
        all_edits = bond_vocab  # Explanation: assigns an intermediate value used by later computation

        cover_num = 0  # Explanation: assigns an intermediate value used by later computation
        for idx, rxn_data in enumerate(rxns_data):  # Explanation: iterates over this collection to process each item
            cover = True  # Explanation: assigns an intermediate value used by later computation
            for edit in rxn_data.edits:  # Explanation: iterates over this collection to process each item
                if edit != 'Terminate' and edit not in all_edits:  # Explanation: checks this condition to choose the next execution path
                    print(f'{edit} in {args.mode} is not in train set')  # Explanation: prints progress or diagnostic information
                    cover = False  # Explanation: assigns an intermediate value used by later computation
            if cover:  # Explanation: checks this condition to choose the next execution path
                cover_num += 1  # Explanation: assigns an intermediate value used by later computation

            counter.append(len(rxn_data.edits))  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies

        print(Counter(counter))  # Explanation: prints progress or diagnostic information
        print(f'The cover rate is {cover_num}/{len(rxns_data)}')  # Explanation: prints progress or diagnostic information
        joblib.dump(rxns_data, save_file, compress=3)  # Explanation: saves processed dataset or vocabulary objects


def main():  # Explanation: defines main, which runs this script from command-line arguments
    parser = argparse.ArgumentParser()  # Explanation: creates command-line argument parser
    parser.add_argument('--dataset', type=str, default='USPTO_50k',  # Explanation: chooses which USPTO dataset split to use
                        help='dataset: USPTO_50k or uspto_full' or 'uspto_mit')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--mode', type=str, default='train',  # Explanation: selects train, valid, or test split
                        help='Type of dataset being prepared: train or valid or test')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--print_every', type=int,  # Explanation: sets logging frequency
                        default=1000, help='Print during preprocessing')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--kekulize', default=True, action='store_true',  # Explanation: controls kekulized molecule preprocessing
                        help='Whether to kekulize mols during training')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--root_dir', type=str, default='.',  # Explanation: selects the root directory containing data and experiments.
                        help='Repository/data root containing data/ and experiments/')  # Explanation: documents the package-relative root directory option.
    args = parser.parse_args()  # Explanation: parses command-line options

    args.dataset = args.dataset.lower()  # Explanation: assigns an intermediate value used by later computation
    paths = resolve_project_paths(args.root_dir)  # Explanation: resolves the root directory used by package CLI file operations.
    datadir = str(paths.dataset_dir(args.dataset))  # Explanation: builds the selected dataset directory from the resolved root.
    rxn_key = 'reactants>reagents>production'  # Explanation: computes an intermediate value for molecular graph editing
    if args.dataset == 'uspto_50k':  # Explanation: checks this condition to choose the next execution path
        filename = f'canonicalized_{args.mode}.csv'  # Explanation: assigns an intermediate value used by later computation
        df = pd.read_csv(os.path.join(datadir, filename))  # Explanation: builds a filesystem path
        preprocessing(rxns=df[rxn_key], args=args,  # Explanation: assigns an intermediate value used by later computation
                      rxn_classes=df['class'], rxns_id=df['id'])  # Explanation: computes an intermediate value for molecular graph editing
    else:  # Explanation: handles the fallback branch for the preceding condition
        filename = f'canonicalized_{args.mode}.csv'  # Explanation: assigns an intermediate value used by later computation
        df = pd.read_csv(os.path.join(datadir, filename))  # Explanation: builds a filesystem path
        preprocessing(rxns=df[rxn_key], args=args)  # Explanation: assigns an intermediate value used by later computation


if __name__ == '__main__':  # Explanation: runs the CLI entry point only when this file is executed directly
    main()  # Explanation: executes this statement as part of extract reaction edit labels and vocabularies
