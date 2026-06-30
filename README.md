# KGCL

## Title

KGCL: Knowledge-Enhanced Graph Contrastive Learning for Retrosynthesis Prediction Based on Molecular Graph Editing

## Environment Requirements

- python = 3.11.8
- pytorch = 2.2.2
- numpy = 1.26.4
- rdkit = 2024.03.4

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

## Reaction-class conditioning

The default setting is without reaction class. Add `--use_rxn_class` to prepare, train, and evaluate with reaction-class conditioning. The flag switches saved data and experiment paths between `without_rxn_class` and `with_rxn_class`.

USPTO-50K supports both reaction-class-unknown evaluation and reaction-class-known evaluation. The USPTO-FULL examples below stay reaction-class-unknown; only use reaction-class conditioning there when you have matching prepared data and a matching checkpoint.

Without reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-prepare-data --dataset uspto_50k --mode valid
kgcl-prepare-data --dataset uspto_50k --mode test
kgcl-train --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k
```

With reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --use_rxn_class
kgcl-prepare-data --dataset uspto_50k --mode valid --use_rxn_class
kgcl-prepare-data --dataset uspto_50k --mode test --use_rxn_class
kgcl-train --dataset uspto_50k --use_rxn_class
kgcl-eval-50k --dataset uspto_50k --use_rxn_class
```

## Model variants

`--model_variant kgcl` is the default and preserves the baseline KGCL implementation. `--model_variant contextual_2fwl` enables the contextual functional-group + sparse 2-FWL-inspired variant. The contextual variant is sparse, local, bridge-closed, task-restricted, and 2-FWL-inspired; it is not full dense 2-FWL over all `V x V` atom pairs.

Accepted names for the contextual variant are:

- `contextual_2fwl`
- `contextual_fg_2fwl`
- `contextual-fg-kgcl-2fwl`

Prepared data for `contextual_2fwl` is not compatible with baseline KGCL prepared shards. Rerun prepare-data for train, valid, and test with the same model variant that you will use for training and evaluation. Evaluation loads model configuration from the checkpoint, so a baseline KGCL checkpoint does not become contextual only because `--model_variant contextual_2fwl` is passed at evaluation time. Evaluate a checkpoint trained with the intended variant, using `--experiments` when needed to select it.

Contextual variant without reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --model_variant contextual_2fwl
kgcl-prepare-data --dataset uspto_50k --mode valid --model_variant contextual_2fwl
kgcl-prepare-data --dataset uspto_50k --mode test --model_variant contextual_2fwl
kgcl-train --dataset uspto_50k --model_variant contextual_2fwl
kgcl-eval-50k --dataset uspto_50k --model_variant contextual_2fwl
```

Contextual variant with reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --use_rxn_class --model_variant contextual_2fwl
kgcl-prepare-data --dataset uspto_50k --mode valid --use_rxn_class --model_variant contextual_2fwl
kgcl-prepare-data --dataset uspto_50k --mode test --use_rxn_class --model_variant contextual_2fwl
kgcl-train --dataset uspto_50k --use_rxn_class --model_variant contextual_2fwl
kgcl-eval-50k --dataset uspto_50k --use_rxn_class --model_variant contextual_2fwl
```

Additional contextual controls exposed by the CLI include `--fg_pool {sum,mean,max}` for contextual functional-group pooling and `--no-pair_use_proposal` to disable proposal scoring.

## Top-k evaluation

USPTO-50K evaluation uses beam search. `kgcl-eval-50k` defaults to `--beam_size 50`, reports cumulative top-k exact-match accuracy for `k = 1, 3, 5, 10, 50`, and reports MaxFrag accuracy for the same k values.

USPTO-FULL evaluation also uses beam search. `kgcl-eval-full` defaults to `--beam_size 10` and reports top-k exact-match accuracy for `k = 1, 3, 5, 10`.

Round-trip evaluation is exposed as `kgcl-eval-roundtrip`. It defaults to USPTO-50K with beam size 50, reads `forward_predictions_50k_top50.txt` from the selected experiment directory, and reports top-k exact-match accuracy plus round-trip accuracy for `k = 1, 3, 5, 10, 50`.

```bash
kgcl-eval-50k --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k --use_rxn_class
kgcl-eval-50k --dataset uspto_50k --beam_size 50

kgcl-eval-full --dataset uspto_full
kgcl-eval-full --dataset uspto_full --beam_size 10

kgcl-eval-roundtrip --dataset uspto_50k
```

## Option reference

| Option                            |         Default | Used by            | Effect                                                                  |
| --------------------------------- | --------------: | ------------------ | ----------------------------------------------------------------------- |
| `--use_rxn_class`                 |             off | prepare/train/eval | Enables reaction-class conditioning and uses `with_rxn_class` paths.    |
| `--model_variant kgcl`            |              on | prepare/train/eval | Uses the baseline KGCL model.                                           |
| `--model_variant contextual_2fwl` |             off | prepare/train/eval | Enables contextual FG + sparse 2-FWL-inspired model.                    |
| `--beam_size 50`                  |  USPTO-50K eval | eval               | Produces candidates for top-k exact-match and MaxFrag at k=1,3,5,10,50. |
| `--beam_size 10`                  | USPTO-FULL eval | eval               | Produces candidates for top-k exact-match at k=1,3,5,10.                |
| `--experiments BEST`              |            BEST | eval               | Selects the experiment/checkpoint directory.                            |
| `--checkpoint epoch_*.pt`         | dataset-specific | eval              | Selects the checkpoint file inside the experiment directory.             |
| `--root_dir .`                    |    current repo | package CLI        | Root containing `data/` and `experiments/`.                             |

## Common recipes

Baseline KGCL, no reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train
kgcl-prepare-data --dataset uspto_50k --mode valid
kgcl-prepare-data --dataset uspto_50k --mode test
kgcl-train --dataset uspto_50k
kgcl-eval-50k --dataset uspto_50k
```

Baseline KGCL, with reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --use_rxn_class
kgcl-prepare-data --dataset uspto_50k --mode valid --use_rxn_class
kgcl-prepare-data --dataset uspto_50k --mode test --use_rxn_class
kgcl-train --dataset uspto_50k --use_rxn_class
kgcl-eval-50k --dataset uspto_50k --use_rxn_class
```

Contextual 2-FWL, no reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --model_variant contextual_2fwl
kgcl-prepare-data --dataset uspto_50k --mode valid --model_variant contextual_2fwl
kgcl-prepare-data --dataset uspto_50k --mode test --model_variant contextual_2fwl
kgcl-train --dataset uspto_50k --model_variant contextual_2fwl
kgcl-eval-50k --dataset uspto_50k --model_variant contextual_2fwl
```

Contextual 2-FWL, with reaction class:

```bash
kgcl-prepare-data --dataset uspto_50k --mode train --use_rxn_class --model_variant contextual_2fwl
kgcl-prepare-data --dataset uspto_50k --mode valid --use_rxn_class --model_variant contextual_2fwl
kgcl-prepare-data --dataset uspto_50k --mode test --use_rxn_class --model_variant contextual_2fwl
kgcl-train --dataset uspto_50k --use_rxn_class --model_variant contextual_2fwl
kgcl-eval-50k --dataset uspto_50k --use_rxn_class --model_variant contextual_2fwl
```

USPTO-FULL evaluation:

```bash
kgcl-prepare-data --dataset uspto_full --mode train
kgcl-prepare-data --dataset uspto_full --mode valid
kgcl-prepare-data --dataset uspto_full --mode test
kgcl-train --dataset uspto_full
kgcl-eval-full --dataset uspto_full
kgcl-eval-full --dataset uspto_full --beam_size 10
```

## Notes and pitfalls

- Run prepare-data for train, valid, and test before training or evaluation.
- Use the same `--use_rxn_class` setting for prepare, train, and eval.
- Use the same `--model_variant` setting for prepare, train, and eval.
- Baseline KGCL and contextual prepared data are not interchangeable.
- Evaluation loads model configuration from the checkpoint, so evaluate a checkpoint trained with the intended variant.
- Do not compare USPTO-50K and USPTO-FULL top-k numbers directly without noting the dataset difference.

## Data

The original datasets used in this paper are from:

USPTO-50K: [https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN) (schneider50k)

USPTO-FULL: [https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN) (uspto_multi)

The raw data and processed data can be accessed via [link](https://drive.google.com/drive/folders/11YMNrm7St-GgVF278orHSXk-EKM3ltqH?usp=sharing). The directory structure should be as follows:

```text
KGCL
|-- data
|   |-- uspto_50k
|   |   |-- canonicalized_test.csv
|   |   |-- canonicalized_train.csv
|   |   |-- canonicalized_val.csv
|   |   |-- raw_test.csv
|   |   |-- raw_train.csv
|   |   `-- raw_val.csv
|   `-- uspto_full
|       |-- canonicalized_test.csv
|       |-- canonicalized_train.csv
|       |-- canonicalized_val.csv
|       |-- raw_test.csv
|       |-- raw_train.csv
|       `-- raw_val.csv
```

- The raw data of the USPTO-50K dataset and USPTO-FULL dataset is stored in the corresponding folders in `raw_train.csv`, `raw_val.csv`, and `raw_test.csv`.
- All processed data files are named `canonicalized_train.csv`, `canonicalized_val.csv`, and `canonicalized_test.csv` and are placed in the corresponding dataset folders.

## Legacy script equivalents

The top-level Python scripts are compatibility wrappers around the package CLI. Prefer package commands for new runs.

```bash
python preprocess.py --mode train --dataset uspto_50k
python preprocess.py --mode valid --dataset uspto_50k
python preprocess.py --mode test --dataset uspto_50k

python prepare_data.py --dataset uspto_50k --mode train
python prepare_data.py --dataset uspto_50k --mode valid
python prepare_data.py --dataset uspto_50k --mode test
python prepare_data.py --dataset uspto_50k --mode train --use_rxn_class
python prepare_data.py --dataset uspto_50k --mode train --model_variant contextual_2fwl

python train.py --dataset uspto_50k
python train.py --dataset uspto_50k --use_rxn_class
python train.py --dataset uspto_50k --model_variant contextual_2fwl

python eval.py --dataset uspto_50k
python eval.py --dataset uspto_50k --use_rxn_class
python eval.py --dataset uspto_50k --beam_size 50
python eval-full.py --dataset uspto_full
python eval-full.py --dataset uspto_full --beam_size 10
python eval-rtacc.py --dataset uspto_50k
```
