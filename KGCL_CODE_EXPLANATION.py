"""
KGCL code explanation and paper mapping.

This file is intentionally documentation-only Python. It mirrors the Markdown
summary in a form that can be opened as source code or run directly:

    python KGCL_CODE_EXPLANATION.py

It does not import RDKit, PyTorch, or project modules, so running it will not
load checkpoints, datasets, or functional-group embeddings.

Paper reviewed:
KGCL: Knowledge-Enhanced Graph Contrastive Learning for Retrosynthesis
Prediction Based on Molecular Graph Editing

Paper file:
/Users/zcahlua/Desktop/future work/KGCL.pdf
"""

PROJECT_PURPOSE = """
KGCL predicts retrosynthetic reactants from a target product molecule by
generating a sequence of molecular graph edits.

The paper's core formulation is:

    product graph -> edit_1 -> intermediate_1 -> ... -> edit_L -> reactant graph

The code follows this formulation through these objects:

    ReactionData -> RxnGraph -> MolGraph -> KGCL -> BeamSearch

The model extends graph-edit retrosynthesis with two main ideas:

1. Functional-group knowledge enhancement:
   - match functional groups in the current molecule using SMARTS patterns;
   - retrieve pretrained functional-group embeddings from fg2emb.pkl;
   - fuse those embeddings into atom features through attention.

2. Graph contrastive learning:
   - use product graph embeddings as anchors;
   - use intermediate graph embeddings from the edit sequence as positives;
   - use other batch pairs as negatives through the pairwise ADNCE loss.
"""

HIGH_LEVEL_PIPELINE = [
    "Canonicalize and atom-map reaction SMILES.",
    "Extract graph edit labels by comparing reactants and products.",
    "Create intermediate product graphs by applying edit labels step by step.",
    "Build MolGraph tensors for each product/intermediate molecule.",
    "Fuse functional-group KG embeddings into atom features when groups match.",
    "Encode the graph with a D-MPNN.",
    "Predict bond-level, atom-level, and graph-level termination edits.",
    "Train with edit cross-entropy plus weighted contrastive loss.",
    "Evaluate by beam-searching edit sequences and comparing final reactants.",
]

PAPER_TO_CODE = [
    {
        "paper_section": "3.1 Problem Definition",
        "concept": "Molecules are graphs; retrosynthesis is an edit sequence.",
        "implementation": [
            "utils/generate_edits.py",
            "utils/reaction_actions.py",
            "prepare_data.py",
        ],
    },
    {
        "paper_section": "3.2 Knowledge-based Molecular Graph Enhancement",
        "concept": "Functional-group matching, KG embedding retrieval, attention fusion.",
        "implementation": [
            "utils/rxn_graphs.py",
            "KGembedding/funcgroup.txt",
            "KGembedding/fg2emb.pkl",
            "KGembedding_2/funcgroup.txt",
            "KGembedding_2/fg2emb.pkl",
        ],
    },
    {
        "paper_section": "3.3 Molecular Graph Encoder",
        "concept": "D-MPNN directed message passing.",
        "implementation": ["models/encoder.py"],
    },
    {
        "paper_section": "3.4 The Generation of Edit Sequences",
        "concept": "Autoregressive atom, bond, and graph edit prediction.",
        "implementation": ["models/KGCL.py", "models/beam_search.py"],
    },
    {
        "paper_section": "3.5 Molecular Graph Contrastive Learning",
        "concept": "Product/intermediate contrastive learning with weighted negatives.",
        "implementation": ["train.py", "utils/ADNCE.py"],
    },
    {
        "paper_section": "3.6 Overall Loss Function",
        "concept": "Edit prediction loss plus lambda-weighted contrastive loss.",
        "implementation": ["train.py"],
    },
    {
        "paper_section": "4 Experiments",
        "concept": "Top-k exact match, MaxFrag, and round-trip evaluation.",
        "implementation": ["eval.py", "eval-full.py", "eval-rtacc.py"],
    },
]

FILE_EXPLANATIONS = {
    "README.md": {
        "role": "Project instructions.",
        "details": [
            "Lists environment requirements.",
            "Explains expected USPTO-50K and USPTO-FULL data layout.",
            "Shows preprocessing, training, and evaluation commands.",
        ],
    },
    "canonicalize_prod.py": {
        "role": "Canonicalizes raw reaction files and remaps atom numbers.",
        "functions": {
            "canonicalize_prod": "Canonicalizes a product and assigns new atom-map numbers.",
            "canonicalize": "Removes hydrogens and atom-map properties, then canonicalizes SMILES.",
            "fix_charge": "Fixes common O, N, and S charge/hydrogen states.",
            "infer_correspondence": "Matches original product atoms to canonicalized product atoms.",
            "remap_rxn_smi": "Applies product atom-map correspondence to reactants.",
            "main": "Reads raw CSV and writes canonicalized CSV.",
        },
        "paper_connection": (
            "Supports graph-edit extraction by making atom maps consistent between "
            "reactants and products."
        ),
    },
    "preprocess.py": {
        "role": "Builds edit labels and edit vocabularies from reactions.",
        "functions": {
            "check_edits": "Rejects reactions requiring Add Bond.",
            "preprocessing": (
                "Parses reactions, filters invalid molecules, calls "
                "generate_reaction_edits, builds vocabularies, and saves ReactionData."
            ),
            "main": "Loads canonicalized CSV files and runs preprocessing.",
        },
        "paper_connection": (
            "Constructs the atom-level, bond-level, and graph-level edit labels "
            "used by KGCL."
        ),
    },
    "prepare_data.py": {
        "role": "Turns ReactionData objects into model-ready training batches.",
        "functions": {
            "apply_edit_to_mol": "Executes an edit tuple on an RDKit molecule.",
            "process_batch": "Creates graph tensors, edit labels, and sequence masks.",
            "prepare_data": (
                "Starts from each product, applies ground-truth edits to create "
                "intermediates, wraps them as RxnGraph objects, and saves batches."
            ),
            "main": "CLI wrapper for batch preparation.",
        },
        "paper_connection": (
            "Creates the edit-sequence training examples for paper Section 3.4 "
            "and the intermediate embeddings used for contrastive learning."
        ),
    },
    "train.py": {
        "role": "Training entry point for KGCL.",
        "functions": {
            "build_model_config": "Creates KGCL config from CLI arguments.",
            "save_checkpoint": "Saves model weights, config, and vocabularies.",
            "train_epoch": (
                "Computes per-step edit cross-entropy, builds product/intermediate "
                "contrastive pairs, applies ADNCE, and updates model parameters."
            ),
            "test": "Greedy validation by comparing predicted edit sequences to labels.",
            "main": "Loads data/vocab, builds model, optimizer, scheduler, and loop.",
        },
        "paper_connection": (
            "Implements Eq. 21 edit loss and Eq. 22 total loss. The code uses "
            "contrastive weights 0.3 without reaction class and 0.4 with reaction class."
        ),
    },
    "eval.py": {
        "role": "USPTO-50K exact-match and MaxFrag evaluation.",
        "functions": {
            "canonicalize_smiles": "Clears atom maps and canonicalizes SMILES.",
            "canonicalize_smiles_clear_map": (
                "Canonicalizes SMILES and optionally returns the largest fragment."
            ),
            "main": "Loads checkpoint, runs BeamSearch, writes predictions, tracks top-k metrics.",
        },
        "paper_connection": "Implements exact match and MaxFrag metrics from Section 4.",
    },
    "eval-full.py": {
        "role": "USPTO-FULL exact-match evaluation.",
        "details": [
            "Similar to eval.py.",
            "Defaults to USPTO-FULL.",
            "Loads epoch_168.pt.",
            "Reports top-k exact match for k = 1, 3, 5, 10.",
        ],
        "paper_connection": "Supports the USPTO-FULL results reported in the paper.",
    },
    "eval-rtacc.py": {
        "role": "Round-trip accuracy evaluation.",
        "functions": {
            "canonicalize": "Clears atom maps and canonicalizes molecules.",
            "canonicalize_p": "Canonicalizes products and assigns atom maps.",
            "smi_tokenizer": "Tokenizes SMILES for forward-model input.",
            "main": (
                "Runs retrosynthesis beam search and compares stored forward-model "
                "predictions to the original product."
            ),
        },
        "paper_connection": "Implements the round-trip accuracy metric from Section 4.",
    },
    "models/KGCL.py": {
        "role": "Main neural model.",
        "class": "KGCL",
        "methods": {
            "_build_layers": (
                "Creates the D-MPNN encoder, optional global attention, and the "
                "three edit-prediction MLPs."
            ),
            "compute_edit_scores": (
                "Encodes one graph state, updates autoregressive atom states, "
                "builds bond and graph features, and outputs flattened edit scores."
            ),
            "forward": "Runs compute_edit_scores over a whole edit sequence.",
            "predict": "Greedy single-molecule edit prediction.",
            "get_saveables": "Returns config and vocabularies for checkpoint reload.",
        },
        "paper_connection": (
            "Implements the autoregressive edit predictor in Eq. 13-18."
        ),
    },
    "models/encoder.py": {
        "role": "Graph encoder implementation.",
        "classes": {
            "MPNEncoder": "Directed message passing neural network.",
            "MultiHeadAttention": "Scaled dot-product multi-head self-attention.",
            "FeedForward": "Residual feed-forward block.",
            "Global_Attention": "Optional self-attention stack over atom embeddings.",
        },
        "paper_connection": "MPNEncoder implements the D-MPNN in Eq. 7-12.",
    },
    "models/beam_search.py": {
        "role": "Beam-search inference over graph edit sequences.",
        "class": "BeamSearch",
        "methods": {
            "process_path": "Expands one partial edit path with top step-level edits.",
            "get_top_k_paths": "Keeps highest-probability paths.",
            "get_edit_from_logits": "Decodes a flattened logit index into an edit action.",
            "run_search": "Generates final top-k reactant candidates.",
        },
        "paper_connection": "Evaluation-time version of autoregressive edit generation.",
    },
    "models/model_utils.py": {
        "role": "Tensor and logging helpers.",
        "functions": {
            "index_select_ND": "Selects message vectors for graph neighborhoods.",
            "creat_edits_feats": "Pads atom embeddings for optional attention.",
            "unbatch_feats": "Converts padded atom embeddings back to flat form.",
            "get_seq_edit_accuracy": "Requires every edit step to be correct.",
            "CSVLogger": "Writes training config and epoch metrics.",
        },
    },
    "utils/rxn_graphs.py": {
        "role": "Molecule/reaction graph representation and functional-group fusion.",
        "functions": {
            "match_fg": (
                "Finds functional groups by SMARTS matching and retrieves KG embeddings."
            ),
            "attention": (
                "Uses atom features as queries and functional-group embeddings as "
                "keys/values; returns atom features plus attended KG information."
            ),
        },
        "classes": {
            "MolGraph": "Builds atom features, fused atom features, directed bond features, and mappings.",
            "RxnGraph": "Stores one product graph plus the edit label for that state.",
            "Vocab": "Maps edit tuples to indices and back.",
        },
        "paper_connection": "Implements functional-group enhancement from Eq. 2-6.",
    },
    "utils/mol_features.py": {
        "role": "Initial chemical feature construction.",
        "functions": {
            "one_of_k_encoding": "One-hot encoding with unknown fallback.",
            "get_atom_features": "Atom symbol, degree, charge, valence, H count, chirality, aromaticity, optional class.",
            "get_bond_features": "Bond type, stereo, conjugation, and ring membership.",
        },
        "paper_connection": "Provides initial atom and bond features before KG enhancement.",
    },
    "utils/generate_edits.py": {
        "role": "Ground-truth graph edit extraction.",
        "functions": {
            "generate_reaction_edits": (
                "Compares mapped reactants/products to identify deleted bonds, "
                "changed bonds, changed atoms, leaving groups, and termination."
            ),
        },
        "paper_connection": "Creates labels for the edit sets Ea, Eb, and Eg.",
    },
    "utils/reaction_actions.py": {
        "role": "Executable graph edit actions.",
        "classes": {
            "ReactionAction": "Abstract base class for edits.",
            "AtomEditAction": "Changes explicit H count and chirality.",
            "BondEditAction": "Deletes or changes a bond and fixes chemistry state.",
            "AddGroupAction": "Attaches a leaving group.",
            "Termination": "Finalizes reactants and handles stereo/chirality.",
        },
        "paper_connection": "Applies predicted graph edits during preprocessing and inference.",
    },
    "utils/chem.py": {
        "role": "RDKit chemistry helper functions.",
        "functions": {
            "get_atom_info": "Atom-map to explicit H and chirality.",
            "get_atom_Chiral": "Atom-map to chiral tag.",
            "get_bond_info": "Atom-map bond pair to bond type/stereo.",
            "get_bond_stereo": "Atom-map bond pair to stereo.",
            "align_kekulize_pairs": "Aligns aromatic/kekulized bond forms.",
            "get_atom_idx": "Finds atom index by atom map.",
            "attach_lg": "Attaches leaving groups at dummy atoms.",
            "fix_Hs_Charge": "Repairs common valence, H, and charge inconsistencies.",
        },
    },
    "utils/collate_fn.py": {
        "role": "Batch tensor and label construction.",
        "functions": {
            "create_pad_tensor": "Pads variable-length graph index arrays.",
            "prepare_edit_labels": (
                "Creates flattened labels in model output order: bond edits, "
                "atom edits, termination."
            ),
            "get_batch_graphs": "Combines MolGraph objects into model input tensors and scopes.",
        },
    },
    "utils/datasets.py": {
        "role": "PyTorch dataset wrappers.",
        "classes": {
            "RetroEditDataset": "Loads saved train batch tensors.",
            "RetroEvalDataset": "Loads processed ReactionData for validation/evaluation.",
        },
    },
    "utils/ADNCE.py": {
        "role": "Adaptive dynamic contrastive loss.",
        "classes": {"ADNCE": "nn.Module wrapper around adnce."},
        "functions": {
            "adnce": (
                "Normalizes embeddings, builds positive/negative logits, applies "
                "Gaussian negative weights, and computes cross-entropy."
            ),
            "transpose": "Transposes final two tensor dimensions.",
            "normalize": "L2-normalizes embeddings.",
        },
        "paper_connection": "Implements the weighted InfoNCE-style Eq. 19-20.",
    },
    "utils/attn_layer.py": {
        "role": "Unused stub file.",
        "details": ["Only imports torch modules; no active implementation."],
    },
}

MODEL_OUTPUT_LAYOUT = """
For each graph state, KGCL.compute_edit_scores returns one flattened vector:

    [all bond edit scores, all atom edit scores, termination score]

utils/collate_fn.prepare_edit_labels builds labels in the same layout:

    [bond_label.flatten(), atom_label.flatten(), stop_label]

This shared layout lets the model use one cross-entropy target over every
possible edit action for the current graph state.
"""

CONTRASTIVE_LEARNING_RECHECK = """
The paper states that contrastive samples are generated from predicted edit
sequences. In this code, train.py receives preprocessed graph sequences from
prepare_data.py. Those sequences were generated by applying extracted
ground-truth edits.

Implementation details:

- batch_graph_outs[0] is used as the product/anchor feature list.
- A random later graph embedding from the same sequence is selected as the
  positive feature when available.
- train.py calls ADNCE(p_features, r_features) without explicit negative_keys.
- ADNCE therefore uses query @ positive_key.T.
- The diagonal entries are positives.
- Off-diagonal entries in the same batch act as negatives.
"""

IMPLEMENTATION_CAVEATS = [
    "train.py has a default dataset value of 'uspto_50k ' with a trailing space; pass --dataset uspto_50k.",
    "eval-rtacc.py writes under pred_text1/{idx}.txt but does not create the pred_text1 directory.",
    "eval.py opens pred_results.txt in append mode, so repeated runs append to the same file.",
    "prepare_data.py may call process_batch on an empty final batch when the dataset size is an exact multiple of batch_size.",
    "Add Bond is filtered by preprocessing and the action implementation does not actually add a missing bond.",
    "utils/rxn_graphs.py overwrites SMARTS globals when loading KGembedding_2; both funcgroup files match in this checkout.",
    "The optional Global_Attention flag is separate from the paper's functional-group attention, which happens in MolGraph.",
]

SOURCE_ASSETS = {
    "KGembedding/funcgroup.txt": "82 functional-group names and SMARTS patterns.",
    "KGembedding/fg2emb.pkl": "Functional-group KG embeddings for no-reaction-class setting.",
    "KGembedding_2/funcgroup.txt": "Second functional-group SMARTS list, same content in this checkout.",
    "KGembedding_2/fg2emb.pkl": "Functional-group KG embeddings for reaction-class setting.",
    "experiments/uspto_50k/without_rxn_class/BEST/epoch_132.pt": "Saved USPTO-50K checkpoint without reaction class.",
    "experiments/uspto_50k/with_rxn_class/BEST/epoch_128.pt": "Saved USPTO-50K checkpoint with reaction class.",
    "experiments/uspto_full/without_rxn_class/BEST/epoch_168.pt": "Saved USPTO-FULL checkpoint.",
    "experiments/uspto_50k/without_rxn_class/BEST/forward_predictions_50k_top50.txt": "Forward-model predictions used for round-trip accuracy.",
}


def _print_header(title):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def _print_bullets(items):
    for item in items:
        print(f"- {item}")


def print_report():
    """Print the KGCL code explanation in a readable terminal format."""
    _print_header("KGCL Code Explanation")
    print(PROJECT_PURPOSE.strip())

    _print_header("High-Level Pipeline")
    for idx, step in enumerate(HIGH_LEVEL_PIPELINE, start=1):
        print(f"{idx}. {step}")

    _print_header("Paper-to-Code Mapping")
    for row in PAPER_TO_CODE:
        print(f"{row['paper_section']}: {row['concept']}")
        print("  Code: " + ", ".join(row["implementation"]))

    _print_header("File Explanations")
    for path, info in FILE_EXPLANATIONS.items():
        print(f"\n{path}")
        print(f"  Role: {info['role']}")
        if "paper_connection" in info:
            print(f"  Paper connection: {info['paper_connection']}")
        for key in ("functions", "classes", "methods"):
            if key in info:
                print(f"  {key.title()}:")
                for name, explanation in info[key].items():
                    print(f"    - {name}: {explanation}")
        if "details" in info:
            print("  Details:")
            for detail in info["details"]:
                print(f"    - {detail}")

    _print_header("Model Output Layout")
    print(MODEL_OUTPUT_LAYOUT.strip())

    _print_header("Contrastive Learning Recheck")
    print(CONTRASTIVE_LEARNING_RECHECK.strip())

    _print_header("Source Assets")
    for path, explanation in SOURCE_ASSETS.items():
        print(f"- {path}: {explanation}")

    _print_header("Implementation Caveats")
    _print_bullets(IMPLEMENTATION_CAVEATS)


if __name__ == "__main__":
    print_report()
