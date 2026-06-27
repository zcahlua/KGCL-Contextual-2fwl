import numpy as np  # Explanation: imports numpy as np for evaluate USPTO-FULL exact-match accuracy
import os  # Explanation: imports os for evaluate USPTO-FULL exact-match accuracy
import argparse  # Explanation: imports argparse for evaluate USPTO-FULL exact-match accuracy
import joblib  # Explanation: imports joblib for evaluate USPTO-FULL exact-match accuracy
from tqdm import tqdm  # Explanation: imports selected names needed to evaluate USPTO-FULL exact-match accuracy
from collections import Counter  # Explanation: imports selected names needed to evaluate USPTO-FULL exact-match accuracy
import torch  # Explanation: imports torch for evaluate USPTO-FULL exact-match accuracy
from rdkit import Chem, RDLogger  # Explanation: imports selected names needed to evaluate USPTO-FULL exact-match accuracy

from kgcl_retro.models import KGCL, BeamSearch  # Explanation: imports packaged model and beam-search classes for evaluation.
from kgcl_retro.config.schema import add_contextual_model_args
from kgcl_retro.paths import resolve_project_paths  # Explanation: imports shared project-root path resolution for package CLIs.
lg = RDLogger.logger()  # Explanation: assigns an intermediate value used by later computation
lg.setLevel(4)  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy

DEFAULT_ROOT_DIR = "."  # Explanation: sets the default root containing data and experiments.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Explanation: defines a module-level constant used by the pipeline

def main():  # Explanation: defines main, which runs this script from command-line arguments
    parser = argparse.ArgumentParser()  # Explanation: creates command-line argument parser
    parser.add_argument('--dataset', type=str, default='USPTO_full',  # Explanation: chooses which USPTO dataset split to use
                        help='dataset: USPTO_50k or USPTO_full')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument("--use_rxn_class", default=False,  # Explanation: enables reaction-class conditioning
                        action='store_true', help='Whether to use rxn_class')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--experiments', type=str, default='BEST',  # Explanation: selects experiment checkpoint directory
                        help='Name of edits prediction experiment')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--beam_size', type=int,  # Explanation: sets evaluation beam width
                        default=10, help='Beam search width')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--max_steps', type=int, default=9,  # Explanation: limits graph edit sequence length
                        help='maximum number of edit steps')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--root_dir', type=str, default=DEFAULT_ROOT_DIR,  # Explanation: selects the root directory containing data and experiments.
                        help='Repository/data root containing data/ and experiments/')  # Explanation: documents the package-relative root directory option.
    add_contextual_model_args(parser)

    args = parser.parse_args()  # Explanation: parses command-line options
    args.dataset = args.dataset.lower()  # Explanation: assigns an intermediate value used by later computation

    paths = resolve_project_paths(args.root_dir)  # Explanation: resolves the root directory used by package CLI file operations.
    data_dir = os.path.join(str(paths.dataset_dir(args.dataset)), 'test')  # Explanation: builds the test split directory from the resolved root.
    test_file = os.path.join(data_dir, 'test.file.kekulized')  # Explanation: builds a filesystem path
    test_data = joblib.load(test_file)  # Explanation: loads processed dataset or vocabulary objects
    if args.use_rxn_class:  # Explanation: checks this condition to choose the next execution path
        exp_dir = os.path.join(  # Explanation: builds a filesystem path
            str(paths.experiments_dir), f'{args.dataset}', 'with_rxn_class', f'{args.experiments}')  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
    else:  # Explanation: handles the fallback branch for the preceding condition
        exp_dir = os.path.join(  # Explanation: builds a filesystem path
            str(paths.experiments_dir), f'{args.dataset}', 'without_rxn_class', f'{args.experiments}')  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy

    checkpoint = torch.load(os.path.join(exp_dir, 'epoch_168.pt'))  # Explanation: loads saved tensor batches or checkpoints
    config = checkpoint['saveables']  # Explanation: assigns an intermediate value used by later computation

    model = KGCL(**config, device=DEVICE)  # Explanation: assigns an intermediate value used by later computation
    model.load_state_dict(checkpoint['state'])  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
    model.to(DEVICE)  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
    model.eval()  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy

    top_k = np.zeros(args.beam_size)  # Explanation: assigns an intermediate value used by later computation
    edit_steps_cor = []  # Explanation: computes an intermediate value for molecular graph editing
    counter = []  # Explanation: assigns an intermediate value used by later computation
    stereo_rxn = []  # Explanation: assigns an intermediate value used by later computation
    stereo_rxn_cor = []  # Explanation: assigns an intermediate value used by later computation
    beam_model = BeamSearch(model=model, step_beam_size=10,  # Explanation: assigns an intermediate value used by later computation
                            beam_size=args.beam_size, use_rxn_class=args.use_rxn_class)  # Explanation: assigns an intermediate value used by later computation
    p_bar = tqdm(list(range(len(test_data))))  # Explanation: assigns an intermediate value used by later computation
    pred_file = os.path.join(exp_dir, 'pred_results.txt')  # Explanation: builds a filesystem path
    file_num = 1  # Explanation: assigns an intermediate value used by later computation
    while os.path.exists(pred_file):  # Explanation: continues looping while the edit-generation condition remains true
        pred_file = os.path.join(exp_dir, f'pred_results_{file_num}.txt')  # Explanation: builds a filesystem path
        file_num += 1  # Explanation: assigns an intermediate value used by later computation

    with open(pred_file, 'a') as fp:  # Explanation: opens a managed resource and closes it automatically
        for idx in p_bar:  # Explanation: iterates over this collection to process each item
            rxn_data = test_data[idx]  # Explanation: computes an intermediate value for molecular graph editing
            rxn_smi = rxn_data.rxn_smi  # Explanation: computes an intermediate value for molecular graph editing
            rxn_class = rxn_data.rxn_class  # Explanation: computes an intermediate value for molecular graph editing
            edit_steps = len(rxn_data.edits)  # Explanation: computes an intermediate value for molecular graph editing
            counter.append(edit_steps)  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy

            r, p = rxn_smi.split('>>')  # Explanation: assigns an intermediate value used by later computation
            r_mol = Chem.MolFromSmiles(r)  # Explanation: parses a SMILES string into an RDKit molecule
            [a.ClearProp('molAtomMapNumber') for a in r_mol.GetAtoms()]  # Explanation: executes this list-comprehension side effect over molecule atoms
            r_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))  # Explanation: parses a SMILES string into an RDKit molecule
            r_smi = Chem.MolToSmiles(r_mol, isomericSmiles=True)  # Explanation: serializes an RDKit molecule back to SMILES
            r_set = set(r_smi.split('.'))  # Explanation: assigns an intermediate value used by later computation

            with torch.no_grad():  # Explanation: opens a managed resource and closes it automatically
                top_k_results = beam_model.run_search(  # Explanation: assigns an intermediate value used by later computation
                    prod_smi=p, max_steps=args.max_steps, rxn_class=rxn_class)  # Explanation: computes an intermediate value for molecular graph editing

            fp.write(f'({idx}) {rxn_smi}\n')  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy

            beam_matched = False  # Explanation: assigns an intermediate value used by later computation
            for beam_idx, path in enumerate(top_k_results):  # Explanation: iterates over this collection to process each item
                pred_smi = path['final_smi']  # Explanation: assigns an intermediate value used by later computation
                prob = path['prob']  # Explanation: assigns an intermediate value used by later computation
                pred_set = set(pred_smi.split('.'))  # Explanation: assigns an intermediate value used by later computation
                correct = pred_set == r_set  # Explanation: assigns an intermediate value used by later computation
                str_edits = '|'.join(f'({str(edit)};{p})'for edit, p in zip(  # Explanation: assigns an intermediate value used by later computation
                    path['rxn_actions'], path['edits_prob']))  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
                fp.write(  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
                    f'{beam_idx} prediction_is_correct:{correct} probability:{prob} {pred_smi} {str_edits}\n')  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
                if correct and not beam_matched:  # Explanation: checks this condition to choose the next execution path
                    top_k[beam_idx] += 1  # Explanation: assigns an intermediate value used by later computation
                    beam_matched = True  # Explanation: assigns an intermediate value used by later computation

            fp.write('\n')  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
            if beam_matched:  # Explanation: checks this condition to choose the next execution path
                edit_steps_cor.append(edit_steps)  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy

            for edit in rxn_data.edits:  # Explanation: iterates over this collection to process each item
                if edit[1] == (1, 1) or edit[1] == (1, 2) or edit[1] == (0, 1) or edit[1] == (0, 2) or edit[1] == (2, 2) or edit[1] == (2, 3):  # Explanation: checks this condition to choose the next execution path
                    stereo_rxn.append(idx)  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
                    if beam_matched:  # Explanation: checks this condition to choose the next execution path
                        stereo_rxn_cor.append(idx)  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy

            msg = 'average score'  # Explanation: assigns an intermediate value used by later computation
            for beam_idx in [1, 3, 5, 10]:  # Explanation: iterates over this collection to process each item
                match_acc = np.sum(top_k[:beam_idx]) / (idx + 1)  # Explanation: assigns an intermediate value used by later computation
                msg += ', t%d: %.3f' % (beam_idx, match_acc)  # Explanation: assigns an intermediate value used by later computation
            p_bar.set_description(msg)  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy

        edit_steps = Counter(counter)  # Explanation: computes an intermediate value for molecular graph editing
        edit_steps_correct = Counter(edit_steps_cor)  # Explanation: computes an intermediate value for molecular graph editing
        fp.write(f'edit_steps_reaction_number:{edit_steps}\n')  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
        fp.write(  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
            f'edit_steps_reaction_prediction_correct:{edit_steps_correct}\n')  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
        fp.write(f'stereo_reaction_idx:{stereo_rxn}\n')  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
        fp.write((f'stereo_reaction_prediction_correct:{stereo_rxn_cor}\n'))  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy


if __name__ == '__main__':  # Explanation: runs the CLI entry point only when this file is executed directly
    main()  # Explanation: executes this statement as part of evaluate USPTO-FULL exact-match accuracy
