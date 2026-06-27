# KGCL Code Summary and Paper Mapping

This document summarizes the current refactored code according to the paper:

**KGCL: Knowledge-Enhanced Graph Contrastive Learning for Retrosynthesis Prediction Based on Molecular Graph Editing**

Paper file reviewed: `/Users/zcahlua/Desktop/future work/KGCL.pdf`

The repository now exposes the KGCL implementation as a reusable Python package under `src/kgcl_retro/`. The old top-level scripts, `models/`, and `utils/` modules remain as compatibility wrappers.

## High-Level Pipeline

KGCL predicts reactants from a product molecule by applying a sequence of graph edits:

```text
product graph -> edit_1 -> intermediate_1 -> ... -> edit_L -> reactant graph
```

The code implements this as:

1. Canonicalize and atom-map reaction SMILES.
2. Extract ground-truth atom, bond, leaving-group, and termination edits.
3. Build `RxnGraph` and `MolGraph` objects for product and intermediate states.
4. Match molecular functional groups and retrieve knowledge-graph embeddings.
5. Fuse functional-group embeddings into atom features through attention.
6. Encode molecular graphs with a directed message passing neural network.
7. Predict edit actions autoregressively with `KGCL`.
8. Train with edit cross-entropy plus ADNCE contrastive loss.
9. Decode candidate edit sequences with beam search during evaluation.

## Package Layout

Reusable code lives under `src/kgcl_retro/`:

```text
src/kgcl_retro/
├── __init__.py
├── paths.py
├── assets/
│   ├── KGembedding/
│   │   ├── funcgroup.txt
│   │   └── fg2emb.pkl
│   └── KGembedding_2/
│       ├── funcgroup.txt
│       └── fg2emb.pkl
├── chemistry/
│   ├── __init__.py
│   ├── actions.py
│   ├── apply.py
│   ├── chem.py
│   ├── edits.py
│   ├── features.py
│   ├── functional_groups.py
│   └── graphs.py
├── cli/
│   ├── canonicalize_products.py
│   ├── preprocess.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── eval_50k.py
│   ├── eval_full.py
│   └── eval_roundtrip.py
├── data/
│   ├── collate.py
│   └── datasets.py
├── losses/
│   └── adnce.py
└── models/
    ├── beam_search.py
    ├── encoder.py
    ├── kgcl.py
    └── utils.py
```

Compatibility wrappers:

- `canonicalize_prod.py`, `preprocess.py`, `prepare_data.py`, `train.py`, `eval.py`, `eval-full.py`, and `eval-rtacc.py` call the package CLI modules.
- `models/*.py` re-export classes and helpers from `kgcl_retro.models`.
- Most `utils/*.py` files re-export helpers from `kgcl_retro.chemistry`, `kgcl_retro.data`, and `kgcl_retro.losses`; `utils/attn_layer.py` remains a tiny unused legacy file.

## Paper-to-Code Mapping

| Paper section | Concept | Primary implementation |
|---|---|---|
| 3.1 Problem Definition | Retrosynthesis as molecular graph edit sequence prediction | `kgcl_retro.chemistry.edits`, `kgcl_retro.chemistry.actions`, `kgcl_retro.cli.prepare_data` |
| 3.2 Knowledge-based Molecular Graph Enhancement | Functional-group matching, KG embedding lookup, atom-feature fusion | `kgcl_retro.chemistry.functional_groups`, `kgcl_retro.chemistry.graphs` |
| 3.3 Molecular Graph Encoder | D-MPNN message passing over directed bonds | `kgcl_retro.models.encoder` |
| 3.4 Generation of Edit Sequences | Autoregressive atom, bond, and graph edit prediction | `kgcl_retro.models.kgcl` |
| 3.5 Contrastive Learning | Product/intermediate embedding contrastive loss | `kgcl_retro.cli.train`, `kgcl_retro.losses.adnce` |
| 3.6 Overall Loss | Edit cross-entropy plus weighted ADNCE loss | `kgcl_retro.cli.train` |
| 4 Experiments | Exact match, MaxFrag, round-trip accuracy | `kgcl_retro.cli.eval_50k`, `kgcl_retro.cli.eval_full`, `kgcl_retro.cli.eval_roundtrip` |

## Main Modules

### `kgcl_retro.chemistry.chem`

Low-level RDKit helpers:

- `get_atom_info()`: extracts explicit hydrogens and chirality tags by atom map number.
- `get_atom_Chiral()`: extracts atom chirality tags.
- `get_bond_info()`: extracts bond type and stereo by mapped atom pair.
- `get_bond_stereo()`: extracts RDKit stereo objects by mapped atom pair.
- `align_kekulize_pairs()`: keeps reactant/product kekulized bond forms aligned.
- `get_atom_idx()`: finds atom index by atom-map number.
- `attach_lg()`: attaches a leaving group molecule to a main molecule.
- `fix_Hs_Charge()`: repairs common hydrogen and charge states after edits.

### `kgcl_retro.chemistry.actions`

Executable graph edit actions:

- `ReactionAction`: abstract edit action base class.
- `AtomEditAction`: changes atom explicit hydrogen count and chirality.
- `BondEditAction`: changes or deletes bonds.
- `AddGroupAction`: attaches a leaving group.
- `Termination`: finalizes the predicted molecule by clearing atom maps and sanitizing.

Paper role: this is the concrete action vocabulary for the graph edit sequence.

### `kgcl_retro.chemistry.apply`

Shared edit dispatcher:

- `apply_edit_to_mol(mol, edit, edit_atom)`: converts an edit tuple into the correct action class and returns the edited RDKit molecule.

This was extracted from the old `prepare_data.py` script so model code no longer depends on a preprocessing script.

### `kgcl_retro.chemistry.edits`

Ground-truth edit extraction:

- `ReactionData`: named tuple storing reaction SMILES, edit list, edit atom list, reaction class, and reaction id.
- `generate_reaction_edits()`: compares mapped reactant/product molecules and emits delete-bond, change-bond, add-bond, change-atom, attach-leaving-group, and terminate labels.

Paper role: constructs the supervised edit sequence described in the problem definition.

### `kgcl_retro.chemistry.features`

Atom and bond featurization:

- `ATOM_SYMBOL_LIST`, `DEGREES`, `FORMAL_CHARGE`, `VALENCE`, `NUM_Hs`, `CHIRALTAG`, `HYBRIDIZATION`.
- `ATOM_FDIM`: base atom feature dimension.
- `BOND_FDIM`: bond feature dimension.
- `one_of_k_encoding()`: converts categorical chemical values into one-hot lists.
- `get_atom_features()`: builds atom vectors, optionally adding reaction class one-hot features.
- `get_bond_features()`: builds bond vectors from type, stereo, conjugation, and ring flags.

#### Atom-Only Features vs. Atom Features with Reaction Class

The code supports two atom-feature modes controlled by `use_rxn_class`.

Atom-only mode:

```text
use_rxn_class = False
atom feature dimension = 85
embedding set = KGembedding
```

Each atom vector contains only local molecular information:

```text
atom symbol
atom degree
formal charge
hybridization
total valence
total hydrogen count
chirality
aromatic flag
```

Reaction-class mode:

```text
use_rxn_class = True
atom feature dimension = 85 + 10 = 95
embedding set = KGembedding_2
```

This mode appends a 10-dimensional one-hot reaction-class vector to every atom feature. The reaction class is global information for the reaction, but the implementation copies it into each atom vector so the graph encoder receives reaction-type context at every atom.

Example:

```text
atom-only feature:
[local atom chemistry, aromatic flag]

atom feature with reaction class:
[local atom chemistry, aromatic flag, reaction-class one-hot]
```

The reaction-class vector is defined by `RXN_CLASSES = list(range(10))`. If the reaction class is `3`, the appended vector is:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

How this is used in the model pipeline:

```text
1. `get_atom_features()` creates either 85-dimensional or 95-dimensional atom vectors.
2. `MolGraph._build_graph()` stores those vectors in `self.f_atoms`.
3. `match_fg()` chooses KG embeddings with the same dimension:
   - `KGembedding` for 85-dimensional atom features.
   - `KGembedding_2` for 95-dimensional atom features.
4. `attention()` fuses matched functional-group embeddings into the atom vectors.
5. Bond features are constructed by concatenating atom features with bond features.
6. `kgcl_retro.data.collate.get_batch_graphs()` batches these atom and bond tensors.
7. `MPNEncoder` consumes tensors whose input dimension is configured from the same `use_rxn_class` setting.
8. `KGCL` predicts graph edit actions from the encoded graph representation.
```

The model configuration mirrors the same choice:

```python
if args.get('use_rxn_class', False):
    atom_fdim = ATOM_FDIM + 10
else:
    atom_fdim = ATOM_FDIM
```

Conceptually, the 85-dimensional version represents an atom-only molecular graph. The 95-dimensional version represents a reaction-class-conditioned molecular graph: each atom still has its own chemical features, but it also carries the known reaction type as context.

### `kgcl_retro.chemistry.functional_groups`

Package-resource loader for KG assets:

- `FunctionalGroupResources`: immutable container for functional-group names, SMARTS queries, lookup maps, and embeddings.
- `load_functional_group_resources(embedding_set)`: loads `funcgroup.txt` and `fg2emb.pkl` from packaged assets.

This replaces the old import-time `open("KGembedding/...")` behavior, so installed package code no longer depends on the current working directory.

### `kgcl_retro.chemistry.graphs`

Molecular graph construction and KG fusion:

- `match_fg()`: matches functional-group SMARTS patterns and returns matched KG embeddings and names.
- `attention()`: fuses atom features with functional-group embeddings.
- `MolGraph`: stores atom features, bond features, atom-to-bond mappings, reverse-bond mappings, and functional-group-enhanced features for one molecule.
- `RxnGraph`: stores the current product graph plus the edit to apply at that step.
- `Vocab`: maps edit tuples to integer ids and back.

Paper role: implements the knowledge-enhanced molecular graph in Section 3.2.

#### Functional-Group Substructure Matching Algorithm

The functional-group matching step is implemented by `match_fg()` in `kgcl_retro.chemistry.graphs`. The code does not implement a custom graph-isomorphism search. Instead, it defines a list of functional-group SMARTS queries and delegates the actual substructure matching to RDKit.

Algorithm used by the code:

```text
1. Choose the functional-group embedding set:
   - `KGembedding` when reaction class is not used.
   - `KGembedding_2` when reaction class is used.
2. Load `funcgroup.txt`, where each row contains:
   - functional-group name
   - SMARTS substructure pattern
3. Convert every SMARTS string into an RDKit query molecule using `Chem.MolFromSmarts()`.
4. For the current product molecule, test each query with `mol.HasSubstructMatch(sm)`.
5. If the molecule contains the queried substructure:
   - recover the functional-group name
   - append the corresponding vector from `fg2emb.pkl`
   - append the functional-group name for diagnostics
6. Return the matched functional-group embedding vectors and names.
```

Core code path:

```python
for sm in resources.smarts:
    if mol.HasSubstructMatch(sm):
        name = resources.smarts_to_name[sm]
        fg_emb.append(resources.embeddings[name].tolist())
        fg_names.append(name)
```

Important behavior:

- `HasSubstructMatch()` only checks whether at least one match exists.
- The implementation records one embedding per matched functional-group type.
- It does not count how many times the functional group appears.
- It does not record which atom indices matched the SMARTS pattern.
- Repeated occurrences of the same functional group still produce only one entry for that functional-group type.

Example SMARTS definitions from `funcgroup.txt`:

```text
Alkyl       [CX4]
Alkenyl     [$([CX3]=[CX3])]
Phenyl      c
Hydroxyl    [#6][OX2H]
Carboxyl    [CX3](=O)[OX2H1]
```

After matching, `MolGraph._build_graph()` stores the matched functional-group vectors in `self.f_fgs`. If `self.f_fgs` is non-empty, the code converts the atom features and functional-group embeddings into tensors and calls `attention()`. This attention step injects the matched functional-group knowledge into every atom feature before bond features are constructed.

In paper terms, this is the code path that maps product molecules to the knowledge-enhanced molecular graph: SMARTS matching decides which chemical knowledge concepts are present, and attention fuses their knowledge-graph embeddings with the atom-level molecular representation.

### `kgcl_retro.data.collate`

Tensor batching:

- `create_pad_tensor()`: pads variable-length neighbor index lists.
- `prepare_edit_labels()`: builds flattened one-hot atom, bond, and graph edit labels.
- `get_batch_graphs()`: converts a list of `MolGraph` objects into tensors consumed by the encoder.

### `kgcl_retro.data.datasets`

Dataset wrappers:

- `RetroEditDataset`: loads saved `batch-{n}.pt` training tensors.
- `RetroEvalDataset`: loads processed `ReactionData` objects for validation or evaluation.

### `kgcl_retro.models.encoder`

Graph encoder:

- `MPNEncoder`: directed message passing network over molecular bonds.
- `MultiHeadAttention`: scaled dot-product attention block.
- `FeedForward`: residual feed-forward layer for attention output.
- `Global_Attention`: optional global attention stack over atom embeddings.

Paper role: corresponds to the molecular graph encoder in Section 3.3.

### `kgcl_retro.models.kgcl`

Main neural retrosynthesis model:

- `KGCL`: predicts atom edits, bond edits, and graph termination scores from encoded molecular graphs.
- `forward()`: scores edit actions for a sequence of graph states.
- `predict()`: greedily predicts an edit sequence for one product.
- `get_saveables()`: stores config and vocabularies needed for checkpoint reload.

Paper role: implements autoregressive edit generation in Section 3.4.

### `kgcl_retro.models.beam_search`

Inference decoder:

- `BeamSearch`: maintains candidate edit paths, scores actions, applies edits, and returns top-k predicted reactants.

Paper role: supports top-k evaluation.

### `kgcl_retro.models.utils`

Model utilities:

- `index_select_ND()`: gathers neighbor/message tensors by index.
- `creat_edits_feats()`: pads atom embeddings for edit attention.
- `unbatch_feats()`: restores padded atom embeddings to flat graph format.
- `get_seq_edit_accuracy()`: computes exact edit-sequence accuracy.
- `CSVLogger`: writes training metrics.

### `kgcl_retro.losses.adnce`

Adaptive debiased contrastive learning:

- `ADNCE`: PyTorch module wrapper.
- `adnce()`: computes contrastive loss over product/intermediate embeddings with Gaussian negative weighting.
- `transpose()` and `normalize()`: tensor helpers for similarity computation.

Paper role: corresponds to the contrastive objective in Sections 3.5 and 3.6.

## CLI Modules

Each package CLI accepts `--root_dir`, which points to the directory containing `data/` and `experiments/`.

Preferred installed commands:

```bash
kgcl-canonicalize --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-train --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k
kgcl-eval-full --dataset uspto_full
kgcl-eval-roundtrip --dataset uspto_50k
```

Legacy script commands still work:

```bash
python canonicalize_prod.py --dataset uspto_50k --mode train
python preprocess.py --dataset uspto_50k --mode train
python prepare_data.py --dataset uspto_50k --mode train
python train.py --dataset uspto_50k
python eval.py --dataset uspto_50k
python eval-full.py --dataset uspto_full
python eval-rtacc.py --dataset uspto_50k
```

## Important Recheck Notes

1. The paper says contrastive samples are generated using predicted edit sequences. This implementation trains contrastive pairs from preprocessed ground-truth product/intermediate graph sequences saved by `kgcl_retro.cli.prepare_data`.
2. `adnce()` is called without explicit `negative_keys`, so in-batch off-diagonal product/intermediate pairs are used as negatives.
3. Functional-group assets are now loaded through `importlib.resources` from `kgcl_retro.assets`, avoiding the old working-directory dependency.
4. `Add Bond` is still rejected during preprocessing by `check_edits()` in `kgcl_retro.cli.preprocess`. The packaging refactor did not change that behavior.
5. `eval_roundtrip.py` still assumes the external forward-model prediction files exist. The package refactor does not replace that external evaluation dependency.
6. The original top-level `train.py` default dataset had a trailing space. The packaged train parser now defaults to `uspto_50k`.
7. All touched Python files include inline `# Explanation:` comments, including the new package files and compatibility wrappers.

## Tests and Verification Files

New tests were added under `tests/`:

- `tests/test_imports.py`: package import and legacy import smoke tests.
- `tests/test_assets.py`: functional-group asset loader behavior.
- `tests/test_edit_labels.py`: edit extraction includes `Terminate`.
- `tests/test_model_config.py`: model export smoke test.

These tests require the project dependencies listed in `pyproject.toml`, especially `torch`, `rdkit`, `numpy`, and `pytest`.

## Plain Explanation of Reaction Class Usage

Reaction class is optional extra information that tells KGCL what type of chemical transformation the reaction belongs to. It is not the same thing as a functional group. A functional group describes a substructure inside a molecule, such as hydroxyl, phenyl, or carboxyl. A reaction class describes the overall transformation type, such as reduction, oxidation, protection, or C-C bond formation.

In this codebase, reaction class is controlled by `use_rxn_class`.

```text
use_rxn_class = False
reaction-class-unknown mode
atom feature dimension = 85
model input = product molecular graph only
```

```text
use_rxn_class = True
reaction-class-known mode
atom feature dimension = 95
model input = product molecular graph + known reaction class
```

The 95-dimensional version is created by appending a 10-dimensional one-hot reaction-class vector to each 85-dimensional atom feature. The reaction class is global reaction information, but the implementation copies the same class vector into every atom feature so the graph encoder receives reaction-type context at every atom.

### What the 10 Reaction Classes Mean

The USPTO-50K dataset uses 10 standard reaction classes. The code stores them only as integer labels, not as text names. The dataset labels are usually `1` to `10`, and the code converts them to model IDs `0` to `9`.

| Dataset class | Model ID | Plain meaning |
|---:|---:|---|
| 1 | 0 | Heteroatom alkylation and arylation: attach a carbon group, such as alkyl or aryl, to a heteroatom like N, O, or S. |
| 2 | 1 | Acylation and related processes: add an acyl group, usually involving a carbonyl group, to another molecule. |
| 3 | 2 | C-C bond formation: create a new carbon-carbon bond to build a larger carbon skeleton. |
| 4 | 3 | Heterocycle formation: form a ring containing at least one non-carbon atom such as N, O, or S. |
| 5 | 4 | Protections: temporarily block a reactive functional group so it does not react in later steps. |
| 6 | 5 | Deprotections: remove a protecting group to recover the original reactive functional group. |
| 7 | 6 | Reductions: decrease oxidation state, often by adding hydrogen or removing oxygen. |
| 8 | 7 | Oxidations: increase oxidation state, often by adding oxygen, removing hydrogen, or increasing bonds to electronegative atoms. |
| 9 | 8 | Functional group interconversion, FGI: change one functional group into another. |
| 10 | 9 | Functional group addition, FGA: add a new functional group to the molecule. |

The code defines the model-side class IDs as:

```python
RXN_CLASSES = list(range(10))
```

During preprocessing, USPTO-50K labels are shifted from one-based labels to zero-based labels:

```python
rxn_class = int(rxn_classes[idx]) - 1
```

So:

```text
dataset class 1  -> model ID 0
dataset class 2  -> model ID 1
...
dataset class 10 -> model ID 9
```

### Where Reaction Class Comes From

In the current KGCL implementation, the reaction class comes from the dataset. For USPTO-50K, the raw CSV contains a `class` column. The canonicalization step keeps this column, and preprocessing stores it inside `ReactionData.rxn_class`.

The reaction class is attached to the whole reaction. It is not recalculated at every graph-edit step. During data preparation, the same stored reaction class is passed to every intermediate `RxnGraph` created from that reaction.

In short:

```text
USPTO-50K CSV class column
-> canonicalized CSV class column
-> ReactionData.rxn_class
-> RxnGraph / MolGraph
-> get_atom_features()
-> 10-dimensional one-hot vector appended to every atom
```

### What KGCL Does Not Do

KGCL does not predict the reaction class first and then use it for retrosynthesis. It also does not infer the reaction class from functional-group matching, and it does not derive the reaction class from previous edit steps.

Current behavior:

```text
reaction class known:
provided by dataset or user

reaction class unknown:
not provided
```

Not current behavior:

```text
product molecule -> KGCL predicts reaction class -> KGCL predicts reactants
```

### Other Possible Ways to Get Reaction Class

Besides reading it from the dataset, a reaction class could be supplied in several ways, but these are extensions rather than current code behavior.

One option is a human chemist or user choosing the class. For example, the user may know that the desired route should be a reduction, oxidation, deprotection, or C-C bond formation.

Another option is a separate reaction-classification model. Such a model could first predict one of the 10 reaction classes from the product molecule or from more reaction context. Then KGCL could use the predicted class as input. This pipeline would look like:

```text
product molecule
-> reaction-classifier model predicts class
-> KGCL uses product + predicted class
-> KGCL predicts reactants
```

A third option is to run KGCL once for each of the 10 classes and then rank all generated predictions:

```text
run KGCL with class 0
run KGCL with class 1
...
run KGCL with class 9
combine and rank predictions
```

This avoids requiring the class beforehand, but it is slower and needs a reliable ranking strategy. The current repository does not implement this all-class reranking pipeline.

A fourth option is rule-based classification from known reactants and products. For example, alcohol to aldehyde is usually oxidation, and ketone to alcohol is usually reduction. However, retrosynthesis inference usually starts with only the product molecule, so this method is useful for labeling or analysis but not enough for ordinary product-only prediction.

The important point is that product structure alone may not uniquely determine reaction class. The same product can often be made by different reaction types. That is why KGCL supports both settings:

```text
reaction-class-known mode:
more information, usually easier, but requires a class label

reaction-class-unknown mode:
less information, usually harder, but more realistic when only the target product is known
```
