from __future__ import division, unicode_literals  # Explanation: imports selected names needed to evaluate round-trip accuracy
import numpy as np  # Explanation: imports numpy as np for evaluate round-trip accuracy
import pandas as pd  # Explanation: imports pandas as pd for evaluate round-trip accuracy
import os  # Explanation: imports os for evaluate round-trip accuracy
import argparse  # Explanation: imports argparse for evaluate round-trip accuracy
import joblib  # Explanation: imports joblib for evaluate round-trip accuracy
from tqdm import tqdm  # Explanation: imports selected names needed to evaluate round-trip accuracy
import torch  # Explanation: imports torch for evaluate round-trip accuracy
from rdkit import Chem, RDLogger  # Explanation: imports selected names needed to evaluate round-trip accuracy

from kgcl_retro.models import KGCL, BeamSearch  # Explanation: imports packaged model and beam-search classes for round-trip evaluation.
from kgcl_retro.paths import resolve_project_paths  # Explanation: imports shared project-root path resolution for package CLIs.

lg = RDLogger.logger()  # Explanation: assigns an intermediate value used by later computation
lg.setLevel(4)  # Explanation: executes this statement as part of evaluate round-trip accuracy

DEFAULT_ROOT_DIR = "."  # Explanation: sets the default root containing data and experiments.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Explanation: defines a module-level constant used by the pipeline


def canonicalize(smi):  # Explanation: defines canonicalize, which removes atom maps/hydrogens and canonicalizes SMILES
    try:  # Explanation: starts a protected block for operations that may fail
        mol = Chem.MolFromSmiles(smi)  # Explanation: parses a SMILES string into an RDKit molecule
    except:  # Explanation: handles failures from the preceding try block
        print('no mol', flush=True)  # Explanation: prints progress or diagnostic information
        return smi  # Explanation: returns this computed result to the caller
    if mol is None:  # Explanation: checks this condition to choose the next execution path
        return smi  # Explanation: returns this computed result to the caller
    mol = Chem.RemoveHs(mol)  # Explanation: removes explicit hydrogen atoms before canonicalization
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms()]  # Explanation: executes this list-comprehension side effect over molecule atoms
    return Chem.MolToSmiles(mol)  # Explanation: returns this computed result to the caller


def canonicalize_p(smi):  # Explanation: defines canonicalize_p, which canonicalizes product SMILES and assigns atom maps
    p = canonicalize(smi)  # Explanation: assigns an intermediate value used by later computation
    p_mol = Chem.MolFromSmiles(p)  # Explanation: parses a SMILES string into an RDKit molecule
    [a.SetAtomMapNum(a.GetIdx() + 1) for a in p_mol.GetAtoms()]  # Explanation: executes this list-comprehension side effect over molecule atoms
    p_smi = Chem.MolToSmiles(p_mol)  # Explanation: serializes an RDKit molecule back to SMILES
    return p_smi  # Explanation: returns this computed result to the caller


def smi_tokenizer(smi):  # Explanation: defines smi_tokenizer, which tokenizes SMILES for the forward model
    """
    Tokenize a SMILES molecule or reaction
    """
    import re  # Explanation: imports re for evaluate round-trip accuracy
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"  # Explanation: assigns an intermediate value used by later computation
    regex = re.compile(pattern)  # Explanation: assigns an intermediate value used by later computation
    tokens = [token for token in regex.findall(smi)]  # Explanation: assigns an intermediate value used by later computation
    assert smi == ''.join(tokens)  # Explanation: checks an invariant expected by the model pipeline
    return ' '.join(tokens)  # Explanation: returns this computed result to the caller


def main():  # Explanation: defines main, which runs this script from command-line arguments
    parser = argparse.ArgumentParser()  # Explanation: creates command-line argument parser
    parser.add_argument('--dataset', type=str, default='USPTO_50k', help='dataset: USPTO_50k or USPTO_full')  # Explanation: chooses which USPTO dataset split to use
    parser.add_argument("--use_rxn_class", default=False,  # Explanation: enables reaction-class conditioning
                        action='store_true', help='Whether to use rxn_class')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--experiments', type=str, default='BEST',  # Explanation: selects experiment checkpoint directory
                        help='Name of edits prediction experiment')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--beam_size', type=int, default=50, help='Beam search width')  # Explanation: sets evaluation beam width
    parser.add_argument('--max_steps', type=int, default=9, help='maximum number of edit steps')  # Explanation: limits graph edit sequence length
    parser.add_argument('--root_dir', type=str, default=DEFAULT_ROOT_DIR,  # Explanation: selects the root directory containing data and experiments.
                        help='Repository/data root containing data/ and experiments/')  # Explanation: documents the package-relative root directory option.

    args = parser.parse_args()  # Explanation: parses command-line options
    args.dataset = args.dataset.lower()  # Explanation: assigns an intermediate value used by later computation

    paths = resolve_project_paths(args.root_dir)  # Explanation: resolves the root directory used by package CLI file operations.
    data_dir = os.path.join(str(paths.dataset_dir(args.dataset)), 'test')  # Explanation: builds the test split directory from the resolved root.
    test_file = os.path.join(data_dir, 'test.file.kekulized')  # Explanation: builds a filesystem path
    test_data = joblib.load(test_file)  # Explanation: loads processed dataset or vocabulary objects
    if args.use_rxn_class:  # Explanation: checks this condition to choose the next execution path
        exp_dir = os.path.join(str(paths.experiments_dir), f'{args.dataset}', 'with_rxn_class', f'{args.experiments}')  # Explanation: builds a filesystem path
    else:  # Explanation: handles the fallback branch for the preceding condition
        exp_dir = os.path.join(str(paths.experiments_dir), f'{args.dataset}', 'without_rxn_class', f'{args.experiments}')  # Explanation: builds a filesystem path

    checkpoint = torch.load(os.path.join(exp_dir, 'epoch_132.pt'))  # Explanation: loads saved tensor batches or checkpoints
    config = checkpoint['saveables']  # Explanation: assigns an intermediate value used by later computation

    model = KGCL(**config, device=DEVICE)  # Explanation: assigns an intermediate value used by later computation
    model.load_state_dict(checkpoint['state'])  # Explanation: executes this statement as part of evaluate round-trip accuracy
    model.to(DEVICE)  # Explanation: executes this statement as part of evaluate round-trip accuracy
    model.eval()  # Explanation: executes this statement as part of evaluate round-trip accuracy

    top_k = np.zeros(args.beam_size)  # Explanation: assigns an intermediate value used by later computation
    rt_top_k = np.zeros(args.beam_size)  # Explanation: assigns an intermediate value used by later computation
    beam_model = BeamSearch(model=model, step_beam_size=10, beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)  # Explanation: assigns an intermediate value used by later computation
    p_bar = tqdm(list(range(len(test_data))))  # Explanation: assigns an intermediate value used by later computation

    # Load the forward model prediction results
    pred_text = os.path.join(exp_dir, 'forward_predictions_50k_top50.txt')  # Explanation: builds a filesystem path
    with open(pred_text, 'r') as f:  # Explanation: opens a managed resource and closes it automatically
        targets = [''.join(line.strip().split(' ')) for line in f.readlines()]  # Explanation: assigns an intermediate value used by later computation

    current_target_idx = 0  # Explanation: assigns an intermediate value used by later computation

    for idx in p_bar:  # Explanation: iterates over this collection to process each item
        rxn_data = test_data[idx]  # Explanation: computes an intermediate value for molecular graph editing
        rxn_smi = rxn_data.rxn_smi  # Explanation: computes an intermediate value for molecular graph editing
        rxn_class = rxn_data.rxn_class  # Explanation: computes an intermediate value for molecular graph editing

        r, p = rxn_smi.split('>>')  # Explanation: assigns an intermediate value used by later computation
        r_mol = Chem.MolFromSmiles(r)  # Explanation: parses a SMILES string into an RDKit molecule
        [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms()]  # Explanation: executes this list-comprehension side effect over molecule atoms
        r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))  # Explanation: parses a SMILES string into an RDKit molecule
        r_smi = Chem.MolToSmiles(r_mol)  # Explanation: serializes an RDKit molecule back to SMILES
        r_set = set(r_smi.split('.'))  # Explanation: assigns an intermediate value used by later computation

        pred_text = os.path.join(exp_dir, 'pred_text1', f'{idx}.txt')  # Explanation: builds a filesystem path

        with torch.no_grad():  # Explanation: opens a managed resource and closes it automatically
            top_k_results = beam_model.run_search(prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class)  # Explanation: assigns an intermediate value used by later computation

        beam_matched = False  # Explanation: assigns an intermediate value used by later computation
        counter = 0  # Explanation: assigns an intermediate value used by later computation
        with open(pred_text, 'a') as fp:  # Explanation: opens a managed resource and closes it automatically
            for beam_idx, path in enumerate(top_k_results):  # Explanation: iterates over this collection to process each item
                pred_smi = path['final_smi']  # Explanation: assigns an intermediate value used by later computation
                prob = path['prob']  # Explanation: assigns an intermediate value used by later computation
                if pred_smi != 'final_smi_unmapped' and prob>0.0:  # Explanation: checks this condition to choose the next execution path
                    pred_result = smi_tokenizer(pred_smi)  # Explanation: assigns an intermediate value used by later computation
                    fp.write(f'{pred_result}\n')  # Explanation: executes this statement as part of evaluate round-trip accuracy
                    counter +=1  # Explanation: assigns an intermediate value used by later computation
                pred_set = set(pred_smi.split('.'))  # Explanation: assigns an intermediate value used by later computation
                if pred_set == r_set and not beam_matched:  # Explanation: checks this condition to choose the next execution path
                    top_k[beam_idx] += 1  # Explanation: assigns an intermediate value used by later computation
                    beam_matched = True  # Explanation: assigns an intermediate value used by later computation
                    true_idx = beam_idx  # Explanation: assigns an intermediate value used by later computation


        p_mol = Chem.MolFromSmiles(p)  # Explanation: parses a SMILES string into an RDKit molecule
        [a.ClearProp('molAtomMapNumber') for a in p_mol.GetAtoms()]  # Explanation: executes this list-comprehension side effect over molecule atoms
        p_mol = Chem.MolFromSmiles(Chem.MolToSmiles(p_mol))  # Explanation: parses a SMILES string into an RDKit molecule
        p_smi = Chem.MolToSmiles(p_mol)  # Explanation: serializes an RDKit molecule back to SMILES

        end_idx = current_target_idx + counter  # Explanation: assigns an intermediate value used by later computation

        all_predictions = list(targets[current_target_idx:end_idx])  # Explanation: assigns an intermediate value used by later computation

        rt_matched = False  # Explanation: assigns an intermediate value used by later computation
        for rt_idx, predictions in enumerate(all_predictions):  # Explanation: iterates over this collection to process each item
            pred = (''.join(predictions.strip().split(' ')))  # Explanation: assigns an intermediate value used by later computation
            mol = Chem.MolFromSmiles(pred)  # Explanation: parses a SMILES string into an RDKit molecule
            if mol is not None:  # Explanation: checks this condition to choose the next execution path
                pred = Chem.MolToSmiles(mol, isomericSmiles=True)  # Explanation: serializes an RDKit molecule back to SMILES
            else:  # Explanation: handles the fallback branch for the preceding condition
                pred = ''  # Explanation: assigns an intermediate value used by later computation

            if beam_matched and not rt_matched:  # Explanation: checks this condition to choose the next execution path
                if true_idx <= rt_idx:  # Explanation: checks this condition to choose the next execution path
                    rt_top_k[true_idx] += 1  # Explanation: assigns an intermediate value used by later computation
                    rt_matched = True  # Explanation: assigns an intermediate value used by later computation
                else:  # Explanation: handles the fallback branch for the preceding condition
                    if pred == p_smi:  # Explanation: checks this condition to choose the next execution path
                        rt_top_k[rt_idx] += 1  # Explanation: assigns an intermediate value used by later computation
                        rt_matched = True  # Explanation: assigns an intermediate value used by later computation
            if pred == p_smi and not beam_matched and not rt_matched:  # Explanation: checks this condition to choose the next execution path
                rt_top_k[rt_idx] += 1  # Explanation: assigns an intermediate value used by later computation
                rt_matched = True  # Explanation: assigns an intermediate value used by later computation

        # update current_target_idx
        current_target_idx = end_idx  # Explanation: assigns an intermediate value used by later computation

        msg = ''  # Explanation: assigns an intermediate value used by later computation
        for beam_idx in [1, 3, 5, 10, 50]:  # Explanation: iterates over this collection to process each item
            match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)  # Explanation: assigns an intermediate value used by later computation
            Rt_acc = np.sum(rt_top_k[:beam_idx]) / (idx + 1)  # Explanation: assigns an intermediate value used by later computation
            msg += 'Exact accuracy, t%d: %.3f' % (beam_idx, match_acc)  # Explanation: assigns an intermediate value used by later computation
            msg += ' Round-trip accuracy, t%d: %.3f ' % (beam_idx, Rt_acc)  # Explanation: assigns an intermediate value used by later computation
        p_bar.set_description(msg)  # Explanation: executes this statement as part of evaluate round-trip accuracy


if __name__ == '__main__':  # Explanation: runs the CLI entry point only when this file is executed directly
    main()  # Explanation: executes this statement as part of evaluate round-trip accuracy
