import argparse  # Explanation: imports argparse for prepare graph-edit training tensors
import copy  # Explanation: imports copy for prepare graph-edit training tensors
import os  # Explanation: imports os for prepare graph-edit training tensors
import sys  # Explanation: imports sys for prepare graph-edit training tensors
from typing import Any  # Explanation: imports generic argument typing used by the preprocessing entry points.

import joblib  # Explanation: imports joblib for prepare graph-edit training tensors
import torch  # Explanation: imports torch for prepare graph-edit training tensors
from rdkit import Chem  # Explanation: imports selected names needed to prepare graph-edit training tensors

from kgcl_retro.chemistry.apply import apply_edit_to_mol  # Explanation: imports the shared edit-application helper for graph sequence generation.
from kgcl_retro.chemistry.actions import Termination  # Explanation: imports the packaged termination action for completed edit sequences.
from kgcl_retro.chemistry.graphs import MolGraph, RxnGraph, Vocab  # Explanation: imports packaged reaction graph and vocabulary helpers.
from kgcl_retro.config.schema import add_contextual_model_args, apply_model_variant_defaults
from kgcl_retro.data.collate import (  # Explanation: imports packaged graph batching and edit-label helpers.
    GraphBatch,
    get_batch_graphs,
    prepare_contextual_edit_labels,
    prepare_edit_labels,
)
from kgcl_retro.paths import resolve_project_paths  # Explanation: imports shared project-root path resolution for package CLIs.

def process_batch(batch_graphs, args):  # Explanation: defines process_batch, which converts graph sequences into padded tensors and labels
    lengths = torch.tensor([len(graph_seq)  # Explanation: assigns an intermediate value used by later computation
                           for graph_seq in batch_graphs], dtype=torch.long)  # Explanation: iterates over this collection to process each item
    max_length = max([len(graph_seq) for graph_seq in batch_graphs])  # Explanation: assigns an intermediate value used by later computation

    paths = resolve_project_paths(getattr(args, 'root_dir', '.'))  # Explanation: resolves the root directory used by package CLI file operations.
    mode_dir = os.path.join(str(paths.dataset_dir(args.dataset)), args.mode)  # Explanation: locates the current dataset split directory from the resolved root.
    bond_vocab_file = os.path.join(mode_dir, 'bond_vocab.txt')  # Explanation: builds the bond vocabulary path for this split.
    atom_vocab_file = os.path.join(mode_dir, 'atom_lg_vocab.txt')  # Explanation: builds the atom and leaving-group vocabulary path for this split.
    bond_vocab = Vocab(joblib.load(bond_vocab_file))  # Explanation: loads processed dataset or vocabulary objects
    atom_vocab = Vocab(joblib.load(atom_vocab_file))  # Explanation: loads processed dataset or vocabulary objects

    graph_seq_tensors = []  # Explanation: computes an intermediate value for molecular graph editing
    edit_seq_labels = []  # Explanation: computes an intermediate value for molecular graph editing
    seq_mask = []  # Explanation: computes an intermediate value for molecular graph editing

    for idx in range(max_length):  # Explanation: iterates over this collection to process each item
        graphs_idx = [copy.deepcopy(batch_graphs[i][min(idx, length-1)]).get_components(attrs=['prod_graph', 'edit_to_apply', 'edit_atom'])  # Explanation: assigns an intermediate value used by later computation
                      for i, length in enumerate(lengths)]  # Explanation: iterates over this collection to process each item
        mask = (idx < lengths).long()  # Explanation: assigns an intermediate value used by later computation
        prod_graphs, edits, edit_atoms = list(zip(*graphs_idx))  # Explanation: computes an intermediate value for molecular graph editing
        assert all([isinstance(graph, MolGraph) for graph in prod_graphs])  # Explanation: checks an invariant expected by the model pipeline

        current_graph_tensors = get_batch_graphs(  # Explanation: assigns an intermediate value used by later computation
            prod_graphs,
            use_rxn_class=args.use_rxn_class,
            model_variant=args.model_variant,
            fg_context_radius=args.fg_context_radius,
            pair_near_radius=args.pair_near_radius,
            pair_bridge_radius=args.pair_bridge_radius,
            pair_max_score_pairs_enc=args.pair_max_score_pairs_enc,
            pair_max_score_pairs_dec=args.pair_max_score_pairs_dec,
            pair_max_carrier_pairs_enc=args.pair_max_carrier_pairs_enc,
            pair_max_carrier_pairs_dec=args.pair_max_carrier_pairs_dec,
            pair_max_bridges_enc=args.pair_max_bridges_enc,
            pair_max_bridges_dec=args.pair_max_bridges_dec,
            pair_topk=args.pair_topk,
        )  # Explanation: computes an intermediate value for molecular graph editing
        if isinstance(current_graph_tensors, GraphBatch):
            edit_labels = prepare_contextual_edit_labels(
                prod_graphs, edits, edit_atoms, bond_vocab, atom_vocab, current_graph_tensors.sparse_metadata
            )
        else:
            edit_labels = prepare_edit_labels(  # Explanation: computes an intermediate value for molecular graph editing
                prod_graphs, edits, edit_atoms, bond_vocab, atom_vocab)  # Explanation: executes this statement as part of prepare graph-edit training tensors

        graph_seq_tensors.append(current_graph_tensors)  # Explanation: executes this statement as part of prepare graph-edit training tensors
        edit_seq_labels.append(edit_labels)  # Explanation: executes this statement as part of prepare graph-edit training tensors
        seq_mask.append(mask)  # Explanation: executes this statement as part of prepare graph-edit training tensors

    seq_mask = torch.stack(seq_mask).long()  # Explanation: stacks tensors along a new dimension
    assert seq_mask.shape[0] == max_length  # Explanation: checks an invariant expected by the model pipeline
    assert seq_mask.shape[1] == len(batch_graphs)  # Explanation: checks an invariant expected by the model pipeline

    return graph_seq_tensors, edit_seq_labels, seq_mask  # Explanation: returns this computed result to the caller


def prepare_data(args: Any) -> None:  # Explanation: defines prepare_data, which creates training batches from processed reactions
    """ 
    prepare data batches for edits prediction
    """
    paths = resolve_project_paths(getattr(args, 'root_dir', '.'))  # Explanation: resolves the root directory used by package CLI file operations.
    mode_dir = os.path.join(str(paths.dataset_dir(args.dataset)), args.mode)  # Explanation: locates the current dataset split directory from the resolved root.
    datafile = os.path.join(mode_dir, f'{args.mode}.file.kekulized')  # Explanation: builds the preprocessed reaction-data input path.
    rxns_data = joblib.load(datafile)  # Explanation: loads processed dataset or vocabulary objects

    batch_graphs = []  # Explanation: assigns an intermediate value used by later computation
    batch_num = 0  # Explanation: assigns an intermediate value used by later computation

    if args.use_rxn_class:  # Explanation: checks this condition to choose the next execution path
        savedir = os.path.join(mode_dir, 'with_rxn_class')  # Explanation: builds the reaction-class-conditioned tensor output directory.
    else:  # Explanation: handles the fallback branch for the preceding condition
        savedir = os.path.join(mode_dir, 'without_rxn_class')  # Explanation: builds the tensor output directory without reaction-class conditioning.
    if args.model_variant != "kgcl":
        savedir = os.path.join(savedir, args.model_variant)
    os.makedirs(savedir, exist_ok=True)  # Explanation: creates output directories when needed

    for idx, rxn_data in enumerate(rxns_data):  # Explanation: iterates over this collection to process each item
        graph_seq = []  # Explanation: computes an intermediate value for molecular graph editing
        rxn_smi = rxn_data.rxn_smi  # Explanation: computes an intermediate value for molecular graph editing
        r, p = rxn_smi.split('>>')  # Explanation: assigns an intermediate value used by later computation
        r_mol = Chem.MolFromSmiles(r)  # Explanation: parses a SMILES string into an RDKit molecule
        p_mol = Chem.MolFromSmiles(p)  # Explanation: parses a SMILES string into an RDKit molecule
        Chem.Kekulize(p_mol)  # Explanation: converts aromatic bonds into kekulized form

        if len(rxn_data.edits) > args.max_steps:  # Explanation: checks this condition to choose the next execution path
            print(f'Edits step exceed max_steps. Skipping reaction {idx}')  # Explanation: prints progress or diagnostic information
            print()  # Explanation: prints progress or diagnostic information
            sys.stdout.flush()  # Explanation: executes this statement as part of prepare graph-edit training tensors
            continue  # Explanation: skips the rest of this loop iteration

        int_mol = p_mol  # Explanation: assigns an intermediate value used by later computation
        for i, edit in enumerate(rxn_data.edits):  # Explanation: iterates over this collection to process each item
            if int_mol is None:  # Explanation: checks this condition to choose the next execution path
                print("Interim mol is None")  # Explanation: prints progress or diagnostic information
                break  # Explanation: exits the current loop early
            if edit == 'Terminate':  # Explanation: checks this condition to choose the next execution path
                graph = RxnGraph(prod_mol=Chem.Mol(  # Explanation: assigns an intermediate value used by later computation
                    int_mol), edit_to_apply=edit, reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=args.use_rxn_class,
                    model_variant=args.model_variant, use_contextual_fg=args.use_contextual_fg,
                    fg_context_radius=args.fg_context_radius, fg_max_instances=args.fg_max_instances,
                    fg_null_token=args.fg_null_token)  # Explanation: assigns an intermediate value used by later computation
                graph_seq.append(graph)  # Explanation: executes this statement as part of prepare graph-edit training tensors
                edit_exe = Termination(action_vocab='Terminate')  # Explanation: computes an intermediate value for molecular graph editing
                try:  # Explanation: starts a protected block for operations that may fail
                    pred_mol = edit_exe.apply(Chem.Mol(int_mol))  # Explanation: assigns an intermediate value used by later computation
                    final_smi = Chem.MolToSmiles(pred_mol)  # Explanation: serializes an RDKit molecule back to SMILES
                except Exception as e:  # Explanation: handles failures from the preceding try block
                    final_smi = None  # Explanation: assigns an intermediate value used by later computation
            else:  # Explanation: handles the fallback branch for the preceding condition
                graph = RxnGraph(prod_mol=Chem.Mol(int_mol), edit_to_apply=edit,  # Explanation: assigns an intermediate value used by later computation
                                 edit_atom=rxn_data.edits_atom[i], reac_mol=Chem.Mol(r_mol), rxn_class=rxn_data.rxn_class, use_rxn_class=args.use_rxn_class,
                                 model_variant=args.model_variant, use_contextual_fg=args.use_contextual_fg,
                                 fg_context_radius=args.fg_context_radius, fg_max_instances=args.fg_max_instances,
                                 fg_null_token=args.fg_null_token)  # Explanation: computes an intermediate value for molecular graph editing
                graph_seq.append(graph)  # Explanation: executes this statement as part of prepare graph-edit training tensors
                int_mol = apply_edit_to_mol(  # Explanation: assigns an intermediate value used by later computation
                    Chem.Mol(int_mol), edit, rxn_data.edits_atom[i])  # Explanation: executes this statement as part of prepare graph-edit training tensors

        if len(graph_seq) == 0 or final_smi is None:  # Explanation: checks this condition to choose the next execution path
            print(f"No valid states found. Skipping reaction {idx}")  # Explanation: prints progress or diagnostic information
            print()  # Explanation: prints progress or diagnostic information
            sys.stdout.flush()  # Explanation: executes this statement as part of prepare graph-edit training tensors
            continue  # Explanation: skips the rest of this loop iteration

        batch_graphs.append(graph_seq)  # Explanation: executes this statement as part of prepare graph-edit training tensors
        if (idx % args.print_every == 0) and idx:  # Explanation: checks this condition to choose the next execution path
            print(f"{idx}/{len(rxns_data)} {args.mode} reactions processed.")  # Explanation: prints progress or diagnostic information
            sys.stdout.flush()  # Explanation: executes this statement as part of prepare graph-edit training tensors

        if (len(batch_graphs) % args.batch_size == 0) and len(batch_graphs):  # Explanation: checks this condition to choose the next execution path
            batch_tensors = process_batch(batch_graphs, args)  # Explanation: assigns an intermediate value used by later computation
            torch.save(batch_tensors, os.path.join(  # Explanation: saves tensor batches or checkpoints
                savedir, f'batch-{batch_num}.pt'))  # Explanation: executes this statement as part of prepare graph-edit training tensors

            batch_num += 1  # Explanation: assigns an intermediate value used by later computation
            batch_graphs = []  # Explanation: assigns an intermediate value used by later computation

    print(f"All {args.mode} reactions complete.")  # Explanation: prints progress or diagnostic information
    sys.stdout.flush()  # Explanation: executes this statement as part of prepare graph-edit training tensors

    batch_tensors = process_batch(batch_graphs, args)  # Explanation: assigns an intermediate value used by later computation
    print("Saving..")  # Explanation: prints progress or diagnostic information
    torch.save(batch_tensors, os.path.join(savedir, f'batch-{batch_num}.pt'))  # Explanation: saves tensor batches or checkpoints


def main():  # Explanation: defines main, which runs this script from command-line arguments
    parser = argparse.ArgumentParser()  # Explanation: creates command-line argument parser
    parser.add_argument('--dataset', type=str, default='USPTO_50k',  # Explanation: chooses which USPTO dataset split to use
                        help='dataset: USPTO_50k or uspto_full or uspto_mit')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--mode', type=str, default='train',  # Explanation: selects train, valid, or test split
                        help='Type of dataset being prepared: train or valid or test')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument("--use_rxn_class", default=False,  # Explanation: enables reaction-class conditioning
                        action='store_true', help='Whether to use rxn_class')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument("--batch_size", default=256,  # Explanation: sets preprocessing batch shard size
                        type=int, help="Number of shards")  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--max_steps', type=int, default=9,  # Explanation: limits graph edit sequence length
                        help='maximum number of edit steps')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--print_every', type=int,  # Explanation: sets logging frequency
                        default=1000, help='Print during preprocessing')  # Explanation: assigns an intermediate value used by later computation
    parser.add_argument('--root_dir', type=str, default='.',  # Explanation: selects the root directory containing data and experiments.
                        help='Repository/data root containing data/ and experiments/')  # Explanation: documents the package-relative root directory option.
    add_contextual_model_args(parser)
    args = parser.parse_args()  # Explanation: parses command-line options

    args.dataset = args.dataset.lower()  # Explanation: assigns an intermediate value used by later computation
    args.__dict__.update(apply_model_variant_defaults(args.__dict__))
    prepare_data(args=args)  # Explanation: assigns an intermediate value used by later computation


if __name__ == "__main__":  # Explanation: runs the CLI entry point only when this file is executed directly
    main()  # Explanation: executes this statement as part of prepare graph-edit training tensors
