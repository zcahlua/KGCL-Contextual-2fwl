# KGCL Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the KGCL research repository into a clean, reusable Python package while preserving the current training, preprocessing, and evaluation behavior.

**Architecture:** Move reusable logic into a `src/kgcl_retro/` package with explicit subpackages for models, chemistry/editing, data, losses, assets, and CLI commands. Keep thin top-level compatibility wrappers so existing README commands can continue to work during the transition. Replace import-time relative asset loading with package-resource asset loading, and add tests that prove the package imports and core tensor/edit utilities work outside the repository root.

**Tech Stack:** Python, PyTorch, RDKit, NumPy, pandas, joblib, tqdm, pytest, setuptools via `pyproject.toml`.

---

## Target File Structure

Create this structure:

```text
pyproject.toml
src/
  kgcl_retro/
    __init__.py
    assets/
      KGembedding/
        funcgroup.txt
        fg2emb.pkl
      KGembedding_2/
        funcgroup.txt
        fg2emb.pkl
    chemistry/
      __init__.py
      actions.py
      chem.py
      edits.py
      features.py
      graphs.py
    cli/
      __init__.py
      canonicalize_products.py
      preprocess.py
      prepare_data.py
      train.py
      eval_50k.py
      eval_full.py
      eval_roundtrip.py
    data/
      __init__.py
      collate.py
      datasets.py
    losses/
      __init__.py
      adnce.py
    models/
      __init__.py
      beam_search.py
      encoder.py
      kgcl.py
      utils.py
    paths.py
tests/
  test_imports.py
  test_assets.py
  test_edit_labels.py
  test_model_config.py
```

Keep these top-level wrappers for backward compatibility:

```text
canonicalize_prod.py
preprocess.py
prepare_data.py
train.py
eval.py
eval-full.py
eval-rtacc.py
models/
utils/
```

Compatibility wrappers should become thin import-and-call files after the package is created. Do not remove existing top-level entry points until all tests and README commands are updated.

---

## Task 1: Add Packaging Metadata

**Files:**
- Create: `pyproject.toml`
- Create: `src/kgcl_retro/__init__.py`
- Test: `tests/test_imports.py`

- [ ] **Step 1: Create the package directory**

Run:

```bash
mkdir -p src/kgcl_retro tests
```

Expected: directories exist.

- [ ] **Step 2: Add `pyproject.toml`**

Create `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "kgcl-retro"
version = "0.1.0"
description = "Knowledge-enhanced graph contrastive learning for retrosynthesis prediction"
readme = "README.md"
requires-python = ">=3.10"
authors = [
  { name = "KGCL maintainers" }
]
dependencies = [
  "joblib",
  "numpy",
  "pandas",
  "torch",
  "tqdm"
]

[project.optional-dependencies]
dev = ["pytest"]
chem = ["rdkit"]

[project.scripts]
kgcl-canonicalize = "kgcl_retro.cli.canonicalize_products:main"
kgcl-preprocess = "kgcl_retro.cli.preprocess:main"
kgcl-prepare-data = "kgcl_retro.cli.prepare_data:main"
kgcl-train = "kgcl_retro.cli.train:main"
kgcl-eval-50k = "kgcl_retro.cli.eval_50k:main"
kgcl-eval-full = "kgcl_retro.cli.eval_full:main"
kgcl-eval-roundtrip = "kgcl_retro.cli.eval_roundtrip:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
kgcl_retro = [
  "assets/KGembedding/funcgroup.txt",
  "assets/KGembedding/fg2emb.pkl",
  "assets/KGembedding_2/funcgroup.txt",
  "assets/KGembedding_2/fg2emb.pkl"
]
```

- [ ] **Step 3: Add package init**

Create `src/kgcl_retro/__init__.py` with:

```python
"""Reusable KGCL retrosynthesis package."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Add import smoke test**

Create `tests/test_imports.py` with:

```python
def test_package_imports():
    import kgcl_retro

    assert kgcl_retro.__version__ == "0.1.0"
```

- [ ] **Step 5: Run the smoke test**

Run:

```bash
PYTHONPATH=src pytest tests/test_imports.py -q
```

Expected: `1 passed`.

Commit:

```bash
git add pyproject.toml src/kgcl_retro/__init__.py tests/test_imports.py
git commit -m "build: add kgcl package metadata"
```

---

## Task 2: Move Assets Behind a Package Resource Loader

**Files:**
- Create: `src/kgcl_retro/assets/KGembedding/funcgroup.txt`
- Create: `src/kgcl_retro/assets/KGembedding/fg2emb.pkl`
- Create: `src/kgcl_retro/assets/KGembedding_2/funcgroup.txt`
- Create: `src/kgcl_retro/assets/KGembedding_2/fg2emb.pkl`
- Create: `src/kgcl_retro/chemistry/functional_groups.py`
- Test: `tests/test_assets.py`

- [ ] **Step 1: Copy asset files into the package**

Run:

```bash
mkdir -p src/kgcl_retro/assets/KGembedding src/kgcl_retro/assets/KGembedding_2
cp KGembedding/funcgroup.txt src/kgcl_retro/assets/KGembedding/funcgroup.txt
cp KGembedding/fg2emb.pkl src/kgcl_retro/assets/KGembedding/fg2emb.pkl
cp KGembedding_2/funcgroup.txt src/kgcl_retro/assets/KGembedding_2/funcgroup.txt
cp KGembedding_2/fg2emb.pkl src/kgcl_retro/assets/KGembedding_2/fg2emb.pkl
```

Expected: four copied package asset files.

- [ ] **Step 2: Add resource loader**

Create `src/kgcl_retro/chemistry/functional_groups.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import pickle
from typing import Any

from rdkit import Chem


@dataclass(frozen=True)
class FunctionalGroupResources:
    names: list[str]
    smarts: list[Any]
    smarts_to_name: dict[Any, str]
    embeddings: dict[str, Any]


def _asset_root(embedding_set: str):
    return resources.files("kgcl_retro").joinpath("assets", embedding_set)


@lru_cache(maxsize=2)
def load_functional_group_resources(embedding_set: str) -> FunctionalGroupResources:
    root = _asset_root(embedding_set)
    funcgroup_text = root.joinpath("funcgroup.txt").read_text()
    rows = [line.split() for line in funcgroup_text.strip().splitlines()]
    names = [row[0] for row in rows]
    smarts = [Chem.MolFromSmarts(row[1]) for row in rows]
    with root.joinpath("fg2emb.pkl").open("rb") as handle:
        embeddings = pickle.load(handle)
    return FunctionalGroupResources(
        names=names,
        smarts=smarts,
        smarts_to_name=dict(zip(smarts, names)),
        embeddings=embeddings,
    )
```

- [ ] **Step 3: Add asset test**

Create `tests/test_assets.py` with:

```python
from kgcl_retro.chemistry.functional_groups import load_functional_group_resources


def test_functional_group_assets_load():
    resources = load_functional_group_resources("KGembedding")

    assert len(resources.names) > 0
    assert len(resources.smarts) == len(resources.names)
    assert "Phenyl" in resources.names
    assert "Phenyl" in resources.embeddings
```

- [ ] **Step 4: Run test**

Run:

```bash
PYTHONPATH=src pytest tests/test_assets.py -q
```

Expected: `1 passed`.

Commit:

```bash
git add src/kgcl_retro/assets src/kgcl_retro/chemistry/functional_groups.py tests/test_assets.py
git commit -m "refactor: load functional group assets as package resources"
```

---

## Task 3: Move Chemistry Utilities Into Package

**Files:**
- Create: `src/kgcl_retro/chemistry/__init__.py`
- Create: `src/kgcl_retro/chemistry/chem.py`
- Create: `src/kgcl_retro/chemistry/actions.py`
- Create: `src/kgcl_retro/chemistry/features.py`
- Modify later wrappers: `utils/chem.py`, `utils/reaction_actions.py`, `utils/mol_features.py`

- [ ] **Step 1: Copy chemistry code**

Run:

```bash
cp utils/chem.py src/kgcl_retro/chemistry/chem.py
cp utils/reaction_actions.py src/kgcl_retro/chemistry/actions.py
cp utils/mol_features.py src/kgcl_retro/chemistry/features.py
```

- [ ] **Step 2: Update imports in copied `actions.py`**

In `src/kgcl_retro/chemistry/actions.py`, replace:

```python
from utils.chem import attach_lg, fix_Hs_Charge, get_atom_Chiral, get_bond_stereo
```

with:

```python
from kgcl_retro.chemistry.chem import attach_lg, fix_Hs_Charge, get_atom_Chiral, get_bond_stereo
```

- [ ] **Step 3: Add package exports**

Create `src/kgcl_retro/chemistry/__init__.py` with:

```python
from kgcl_retro.chemistry.actions import (
    AddGroupAction,
    AtomEditAction,
    BondEditAction,
    ReactionAction,
    Termination,
)
from kgcl_retro.chemistry.features import (
    ATOM_FDIM,
    BOND_FDIM,
    get_atom_features,
    get_bond_features,
)

__all__ = [
    "AddGroupAction",
    "AtomEditAction",
    "BondEditAction",
    "ReactionAction",
    "Termination",
    "ATOM_FDIM",
    "BOND_FDIM",
    "get_atom_features",
    "get_bond_features",
]
```

- [ ] **Step 4: Run import test**

Append to `tests/test_imports.py`:

```python
def test_chemistry_imports():
    from kgcl_retro.chemistry import ATOM_FDIM, BOND_FDIM, BondEditAction

    assert ATOM_FDIM > 0
    assert BOND_FDIM > 0
    assert BondEditAction.__name__ == "BondEditAction"
```

Run:

```bash
PYTHONPATH=src pytest tests/test_imports.py -q
```

Expected: all tests pass.

Commit:

```bash
git add src/kgcl_retro/chemistry tests/test_imports.py
git commit -m "refactor: move chemistry utilities into package"
```

---

## Task 4: Move Edit Extraction and Graph Construction

**Files:**
- Create: `src/kgcl_retro/chemistry/edits.py`
- Create: `src/kgcl_retro/chemistry/graphs.py`
- Modify: `src/kgcl_retro/chemistry/graphs.py`
- Test: `tests/test_edit_labels.py`

- [ ] **Step 1: Copy graph/edit modules**

Run:

```bash
cp utils/generate_edits.py src/kgcl_retro/chemistry/edits.py
cp utils/rxn_graphs.py src/kgcl_retro/chemistry/graphs.py
```

- [ ] **Step 2: Update imports in `edits.py`**

Replace:

```python
from utils.reaction_actions import (AddGroupAction, AtomEditAction,
                                    BondEditAction, Termination)
from utils.chem import align_kekulize_pairs, get_atom_info, get_bond_info
```

with:

```python
from kgcl_retro.chemistry.actions import (
    AddGroupAction,
    AtomEditAction,
    BondEditAction,
    Termination,
)
from kgcl_retro.chemistry.chem import align_kekulize_pairs, get_atom_info, get_bond_info
```

- [ ] **Step 3: Update imports and asset loading in `graphs.py`**

Replace the top-level `open(...)` and `pickle.load(...)` asset-loading block with:

```python
from kgcl_retro.chemistry.features import get_atom_features, get_bond_features
from kgcl_retro.chemistry.functional_groups import load_functional_group_resources
```

Replace `match_fg()` with:

```python
def match_fg(mol, use_rxn_class):
    embedding_set = "KGembedding_2" if use_rxn_class else "KGembedding"
    resources = load_functional_group_resources(embedding_set)
    fg_emb = []
    fg_names = []
    for sm in resources.smarts:
        if mol.HasSubstructMatch(sm):
            name = resources.smarts_to_name[sm]
            fg_emb.append(resources.embeddings[name].tolist())
            fg_names.append(name)
    return fg_emb, fg_names
```

- [ ] **Step 4: Add edit extraction smoke test**

Create `tests/test_edit_labels.py` with:

```python
from kgcl_retro.chemistry.edits import generate_reaction_edits


def test_generate_reaction_edits_adds_termination():
    reaction = "[CH3:1][OH:2]>>[CH3:1][OH:2]"
    data = generate_reaction_edits(reaction, kekulize=False)

    assert data is not None
    assert data.edits[-1] == "Terminate"
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_assets.py tests/test_edit_labels.py -q
```

Expected: tests pass.

Commit:

```bash
git add src/kgcl_retro/chemistry/edits.py src/kgcl_retro/chemistry/graphs.py tests/test_edit_labels.py
git commit -m "refactor: move graph and edit extraction into package"
```

---

## Task 5: Move Data Utilities and Losses

**Files:**
- Create: `src/kgcl_retro/data/__init__.py`
- Create: `src/kgcl_retro/data/collate.py`
- Create: `src/kgcl_retro/data/datasets.py`
- Create: `src/kgcl_retro/losses/__init__.py`
- Create: `src/kgcl_retro/losses/adnce.py`

- [ ] **Step 1: Copy files**

Run:

```bash
mkdir -p src/kgcl_retro/data src/kgcl_retro/losses
cp utils/collate_fn.py src/kgcl_retro/data/collate.py
cp utils/datasets.py src/kgcl_retro/data/datasets.py
cp utils/ADNCE.py src/kgcl_retro/losses/adnce.py
```

- [ ] **Step 2: Update imports in `collate.py`**

Replace:

```python
from utils.mol_features import ATOM_FDIM, BOND_FDIM
from utils.rxn_graphs import MolGraph
```

with:

```python
from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.chemistry.graphs import MolGraph
```

- [ ] **Step 3: Update imports in `datasets.py`**

Replace:

```python
from utils.generate_edits import ReactionData
```

with:

```python
from kgcl_retro.chemistry.edits import ReactionData
```

- [ ] **Step 4: Add exports**

Create `src/kgcl_retro/data/__init__.py` with:

```python
from kgcl_retro.data.collate import get_batch_graphs, prepare_edit_labels
from kgcl_retro.data.datasets import RetroEditDataset, RetroEvalDataset

__all__ = [
    "get_batch_graphs",
    "prepare_edit_labels",
    "RetroEditDataset",
    "RetroEvalDataset",
]
```

Create `src/kgcl_retro/losses/__init__.py` with:

```python
from kgcl_retro.losses.adnce import ADNCE, adnce

__all__ = ["ADNCE", "adnce"]
```

- [ ] **Step 5: Run import tests**

Append to `tests/test_imports.py`:

```python
def test_data_and_loss_imports():
    from kgcl_retro.data import RetroEditDataset, get_batch_graphs
    from kgcl_retro.losses import ADNCE

    assert RetroEditDataset.__name__ == "RetroEditDataset"
    assert callable(get_batch_graphs)
    assert ADNCE.__name__ == "ADNCE"
```

Run:

```bash
PYTHONPATH=src pytest tests/test_imports.py -q
```

Expected: tests pass.

Commit:

```bash
git add src/kgcl_retro/data src/kgcl_retro/losses tests/test_imports.py
git commit -m "refactor: move data utilities and losses into package"
```

---

## Task 6: Move Model Code and Break Script Coupling

**Files:**
- Create: `src/kgcl_retro/models/__init__.py`
- Create: `src/kgcl_retro/models/kgcl.py`
- Create: `src/kgcl_retro/models/beam_search.py`
- Create: `src/kgcl_retro/models/encoder.py`
- Create: `src/kgcl_retro/models/utils.py`
- Create: `src/kgcl_retro/chemistry/apply.py`
- Test: `tests/test_model_config.py`

- [ ] **Step 1: Copy model files**

Run:

```bash
mkdir -p src/kgcl_retro/models
cp models/KGCL.py src/kgcl_retro/models/kgcl.py
cp models/beam_search.py src/kgcl_retro/models/beam_search.py
cp models/encoder.py src/kgcl_retro/models/encoder.py
cp models/model_utils.py src/kgcl_retro/models/utils.py
```

- [ ] **Step 2: Extract edit application helper**

Create `src/kgcl_retro/chemistry/apply.py` with:

```python
from __future__ import annotations

from typing import Any, Tuple

from rdkit import Chem

from kgcl_retro.chemistry.actions import (
    AddGroupAction,
    AtomEditAction,
    BondEditAction,
)


def apply_edit_to_mol(mol: Chem.Mol, edit: Tuple, edit_atom: Any) -> Chem.Mol:
    if edit[0] == "Change Atom":
        return AtomEditAction(edit_atom, *edit[1], action_vocab="Change Atom").apply(mol)
    if edit[0] == "Delete Bond":
        return BondEditAction(*edit_atom, *edit[1], action_vocab="Delete Bond").apply(mol)
    if edit[0] == "Change Bond":
        return BondEditAction(*edit_atom, *edit[1], action_vocab="Change Bond").apply(mol)
    if edit[0] == "Add Bond":
        return BondEditAction(*edit_atom, *edit[1], action_vocab="Add Bond").apply(mol)
    if edit[0] == "Attaching LG":
        return AddGroupAction(edit_atom, edit[1], action_vocab="Attaching LG").apply(mol)
    raise ValueError(f"Unsupported edit action: {edit}")
```

- [ ] **Step 3: Update imports in copied model files**

In `src/kgcl_retro/models/kgcl.py`, replace:

```python
from prepare_data import apply_edit_to_mol
from utils.collate_fn import get_batch_graphs
from utils.rxn_graphs import MolGraph, Vocab
from models.encoder import Global_Attention, MPNEncoder
from models.model_utils import (creat_edits_feats, index_select_ND,
                                unbatch_feats)
```

with:

```python
from kgcl_retro.chemistry.apply import apply_edit_to_mol
from kgcl_retro.chemistry.graphs import MolGraph, Vocab
from kgcl_retro.data.collate import get_batch_graphs
from kgcl_retro.models.encoder import Global_Attention, MPNEncoder
from kgcl_retro.models.utils import creat_edits_feats, index_select_ND, unbatch_feats
```

In `src/kgcl_retro/models/beam_search.py`, replace:

```python
from utils.rxn_graphs import MolGraph
from utils.collate_fn import get_batch_graphs
from prepare_data import apply_edit_to_mol
from utils.reaction_actions import (AddGroupAction, AtomEditAction,
                                    BondEditAction, Termination)
```

with:

```python
from kgcl_retro.chemistry.apply import apply_edit_to_mol
from kgcl_retro.chemistry.actions import AddGroupAction, AtomEditAction, BondEditAction, Termination
from kgcl_retro.chemistry.graphs import MolGraph
from kgcl_retro.data.collate import get_batch_graphs
```

In `src/kgcl_retro/models/encoder.py`, replace:

```python
from models.model_utils import index_select_ND
```

with:

```python
from kgcl_retro.models.utils import index_select_ND
```

- [ ] **Step 4: Add model exports**

Create `src/kgcl_retro/models/__init__.py` with:

```python
from kgcl_retro.models.beam_search import BeamSearch
from kgcl_retro.models.kgcl import KGCL

__all__ = ["BeamSearch", "KGCL"]
```

- [ ] **Step 5: Add model config import test**

Create `tests/test_model_config.py` with:

```python
from kgcl_retro.models import BeamSearch, KGCL


def test_model_classes_import():
    assert KGCL.__name__ == "KGCL"
    assert BeamSearch.__name__ == "BeamSearch"
```

Run:

```bash
PYTHONPATH=src pytest tests/test_model_config.py -q
```

Expected: `1 passed`.

Commit:

```bash
git add src/kgcl_retro/models src/kgcl_retro/chemistry/apply.py tests/test_model_config.py
git commit -m "refactor: move model code into package"
```

---

## Task 7: Move CLI Scripts Into Package and Keep Wrappers

**Files:**
- Create: `src/kgcl_retro/cli/__init__.py`
- Create: `src/kgcl_retro/cli/canonicalize_products.py`
- Create: `src/kgcl_retro/cli/preprocess.py`
- Create: `src/kgcl_retro/cli/prepare_data.py`
- Create: `src/kgcl_retro/cli/train.py`
- Create: `src/kgcl_retro/cli/eval_50k.py`
- Create: `src/kgcl_retro/cli/eval_full.py`
- Create: `src/kgcl_retro/cli/eval_roundtrip.py`
- Modify: top-level `canonicalize_prod.py`, `preprocess.py`, `prepare_data.py`, `train.py`, `eval.py`, `eval-full.py`, `eval-rtacc.py`

- [ ] **Step 1: Copy top-level scripts into CLI package**

Run:

```bash
mkdir -p src/kgcl_retro/cli
cp canonicalize_prod.py src/kgcl_retro/cli/canonicalize_products.py
cp preprocess.py src/kgcl_retro/cli/preprocess.py
cp prepare_data.py src/kgcl_retro/cli/prepare_data.py
cp train.py src/kgcl_retro/cli/train.py
cp eval.py src/kgcl_retro/cli/eval_50k.py
cp eval-full.py src/kgcl_retro/cli/eval_full.py
cp eval-rtacc.py src/kgcl_retro/cli/eval_roundtrip.py
touch src/kgcl_retro/cli/__init__.py
```

- [ ] **Step 2: Update imports in CLI copies**

In each copied CLI file, replace old imports with package imports:

```python
from kgcl_retro.models import KGCL, BeamSearch
from kgcl_retro.models.utils import CSVLogger, get_seq_edit_accuracy
from kgcl_retro.data.datasets import RetroEditDataset, RetroEvalDataset
from kgcl_retro.chemistry.features import ATOM_FDIM, BOND_FDIM
from kgcl_retro.chemistry.graphs import Vocab, MolGraph, RxnGraph
from kgcl_retro.chemistry.edits import generate_reaction_edits
from kgcl_retro.chemistry.apply import apply_edit_to_mol
from kgcl_retro.data.collate import get_batch_graphs, prepare_edit_labels
from kgcl_retro.losses import ADNCE, adnce
```

Only keep imports each file actually uses.

- [ ] **Step 3: Replace top-level wrappers**

Replace top-level `train.py` with:

```python
from kgcl_retro.cli.train import main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Keep the full parser in kgcl_retro.cli.train during this task.
    # This wrapper exists only for backward compatibility.
    main(parser.parse_args().__dict__)
```

For scripts whose `main()` parses its own arguments, use:

```python
from kgcl_retro.cli.preprocess import main


if __name__ == "__main__":
    main()
```

Apply the same pattern to:

```text
canonicalize_prod.py
preprocess.py
prepare_data.py
eval.py
eval-full.py
eval-rtacc.py
```

If `train.py` currently builds its parser under `if __name__ == "__main__"`, keep that parser in `kgcl_retro.cli.train` by moving the bottom block into a `cli_main()` function and making both the console script and wrapper call `cli_main()`.

- [ ] **Step 4: Run CLI import smoke tests**

Run:

```bash
PYTHONPATH=src python -m kgcl_retro.cli.train --help
PYTHONPATH=src python -m kgcl_retro.cli.preprocess --help
PYTHONPATH=src python -m kgcl_retro.cli.eval_50k --help
```

Expected: each command prints help and exits 0.

Commit:

```bash
git add src/kgcl_retro/cli canonicalize_prod.py preprocess.py prepare_data.py train.py eval.py eval-full.py eval-rtacc.py
git commit -m "refactor: move command line entry points into package"
```

---

## Task 8: Add Central Path Handling

**Files:**
- Create: `src/kgcl_retro/paths.py`
- Modify: CLI files under `src/kgcl_retro/cli/`

- [ ] **Step 1: Add path helper**

Create `src/kgcl_retro/paths.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    def dataset_dir(self, dataset: str) -> Path:
        return self.data_dir / dataset


def resolve_project_paths(root: str | Path = ".") -> ProjectPaths:
    return ProjectPaths(root=Path(root).resolve())
```

- [ ] **Step 2: Add `--root_dir` to CLI parsers**

For every CLI parser, add:

```python
parser.add_argument(
    "--root_dir",
    type=str,
    default=".",
    help="Repository/data root containing data/ and experiments/",
)
```

- [ ] **Step 3: Replace global `ROOT_DIR = './'` usage**

In CLI files, replace:

```python
ROOT_DIR = './'
```

and `os.path.join(ROOT_DIR, ...)` paths with:

```python
from kgcl_retro.paths import resolve_project_paths

paths = resolve_project_paths(args.root_dir)
data_dir = paths.dataset_dir(args.dataset)
exp_dir = paths.experiments_dir / args.dataset / "without_rxn_class" / args.experiments
```

Use `Path` objects or convert with `str(...)` only when an API requires a string.

- [ ] **Step 4: Run help smoke tests**

Run:

```bash
PYTHONPATH=src python -m kgcl_retro.cli.train --help | grep root_dir
PYTHONPATH=src python -m kgcl_retro.cli.eval_50k --help | grep root_dir
```

Expected: both commands print `root_dir`.

Commit:

```bash
git add src/kgcl_retro/paths.py src/kgcl_retro/cli
git commit -m "refactor: centralize project path handling"
```

---

## Task 9: Convert Old `models/` and `utils/` to Compatibility Wrappers

**Files:**
- Modify: `models/*.py`
- Modify: `utils/*.py`

- [ ] **Step 1: Replace `models/__init__.py`**

Use:

```python
from kgcl_retro.models import BeamSearch, KGCL

__all__ = ["BeamSearch", "KGCL"]
```

- [ ] **Step 2: Replace model wrapper modules**

Replace `models/KGCL.py` with:

```python
from kgcl_retro.models.kgcl import KGCL

__all__ = ["KGCL"]
```

Replace `models/beam_search.py` with:

```python
from kgcl_retro.models.beam_search import BeamSearch

__all__ = ["BeamSearch"]
```

Replace `models/encoder.py` with:

```python
from kgcl_retro.models.encoder import FeedForward, Global_Attention, MPNEncoder, MultiHeadAttention

__all__ = ["FeedForward", "Global_Attention", "MPNEncoder", "MultiHeadAttention"]
```

Replace `models/model_utils.py` with:

```python
from kgcl_retro.models.utils import CSVLogger, creat_edits_feats, get_seq_edit_accuracy, index_select_ND, unbatch_feats

__all__ = [
    "CSVLogger",
    "creat_edits_feats",
    "get_seq_edit_accuracy",
    "index_select_ND",
    "unbatch_feats",
]
```

- [ ] **Step 3: Replace utility wrapper modules**

For each old utility module, import from the new package module:

```python
# utils/chem.py
from kgcl_retro.chemistry.chem import *
```

```python
# utils/reaction_actions.py
from kgcl_retro.chemistry.actions import *
```

```python
# utils/mol_features.py
from kgcl_retro.chemistry.features import *
```

```python
# utils/rxn_graphs.py
from kgcl_retro.chemistry.graphs import *
```

```python
# utils/generate_edits.py
from kgcl_retro.chemistry.edits import *
```

```python
# utils/collate_fn.py
from kgcl_retro.data.collate import *
```

```python
# utils/datasets.py
from kgcl_retro.data.datasets import *
```

```python
# utils/ADNCE.py
from kgcl_retro.losses.adnce import *
```

- [ ] **Step 4: Run old-import and new-import tests**

Append to `tests/test_imports.py`:

```python
def test_legacy_imports_still_work():
    from models import KGCL
    from utils.rxn_graphs import MolGraph

    assert KGCL.__name__ == "KGCL"
    assert MolGraph.__name__ == "MolGraph"
```

Run:

```bash
PYTHONPATH=src:. pytest tests/test_imports.py -q
```

Expected: all import tests pass.

Commit:

```bash
git add models utils tests/test_imports.py
git commit -m "refactor: make legacy modules compatibility wrappers"
```

---

## Task 10: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `KGCL_CODE_SUMMARY.md`

- [ ] **Step 1: Add installation instructions to README**

Add:

```markdown
## Install as a Package

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

If RDKit is not available through pip in your environment, install RDKit with conda first:

```bash
conda install -c conda-forge rdkit
python -m pip install -e ".[dev]"
```
```

- [ ] **Step 2: Add package CLI commands to README**

Add:

```markdown
## Package CLI

The legacy scripts still work, but the package commands are preferred:

```bash
kgcl-canonicalize --dataset uspto_50k --mode train
kgcl-preprocess --dataset uspto_50k --mode train
kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-train --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k
kgcl-eval-full --dataset uspto_full
kgcl-eval-roundtrip --dataset uspto_50k
```
```

- [ ] **Step 3: Update architecture summary**

In `KGCL_CODE_SUMMARY.md`, add a section:

```markdown
## Package Layout

Reusable code now lives under `src/kgcl_retro/`:

- `kgcl_retro.models`: KGCL, D-MPNN encoder, beam search, model utilities.
- `kgcl_retro.chemistry`: chemistry helpers, edit actions, graph construction, functional-group assets.
- `kgcl_retro.data`: datasets and collate functions.
- `kgcl_retro.losses`: ADNCE contrastive loss.
- `kgcl_retro.cli`: command-line entry points.

Top-level scripts and old `models/`/`utils/` modules are compatibility wrappers.
```

- [ ] **Step 4: Run documentation grep checks**

Run:

```bash
grep -n "Install as a Package" README.md
grep -n "Package CLI" README.md
grep -n "Package Layout" KGCL_CODE_SUMMARY.md
```

Expected: all three headings are found.

Commit:

```bash
git add README.md KGCL_CODE_SUMMARY.md
git commit -m "docs: document reusable package layout"
```

---

## Task 11: Final Verification

**Files:**
- No new files.
- Verify full package and legacy compatibility.

- [ ] **Step 1: Run full syntax compile**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/kgcl_pycache python -m py_compile $(find src models utils -name '*.py') canonicalize_prod.py preprocess.py prepare_data.py train.py eval.py eval-full.py eval-rtacc.py
```

Expected: exit code 0.

- [ ] **Step 2: Run test suite**

Run:

```bash
PYTHONPATH=src:. pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Verify console entry points after editable install**

Run:

```bash
python -m pip install -e ".[dev]"
kgcl-train --help
kgcl-preprocess --help
kgcl-eval-50k --help
```

Expected: each command prints help and exits 0.

- [ ] **Step 4: Verify legacy scripts still expose help**

Run:

```bash
python train.py --help
python preprocess.py --help
python eval.py --help
```

Expected: each command prints help and exits 0.

- [ ] **Step 5: Commit final verification-only changes if any**

If verification required a small fix:

```bash
git add <fixed-files>
git commit -m "test: verify package refactor"
```

If no files changed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Reusable package metadata: Task 1.
- Package assets instead of repo-root-relative assets: Task 2.
- Chemistry/edit utilities modularized: Tasks 3 and 4.
- Data and loss utilities modularized: Task 5.
- Model code modularized: Task 6.
- Script entry points moved into package CLI: Task 7.
- Root/data/experiment path handling cleaned up: Task 8.
- Backward compatibility for old imports and scripts: Task 9.
- Documentation updated: Task 10.
- Full verification: Task 11.

Known deliberate constraints:

- This plan preserves behavior first. It does not redesign the model, training loss, or preprocessing algorithm.
- Existing inline `# Explanation:` comments are not removed by this plan. If clean source style is required, add a separate task after packaging to move those explanations into docs and keep only useful comments in code.
- The existing `Add Bond` limitation is not fixed in this packaging refactor. It should be a separate behavioral task with chemistry tests.

