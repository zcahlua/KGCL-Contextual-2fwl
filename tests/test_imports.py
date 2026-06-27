def test_package_imports():  # Explanation: verifies the root package exposes version metadata.
    import kgcl_retro  # Explanation: imports the package without touching heavy chemistry or model dependencies.

    assert kgcl_retro.__version__ == "0.1.0"  # Explanation: checks that package metadata matches pyproject.toml.


def test_chemistry_imports():  # Explanation: verifies the public chemistry API is available when dependencies are installed.
    from kgcl_retro.chemistry import ATOM_FDIM, BOND_FDIM, BondEditAction  # Explanation: imports feature dimensions and an edit action from the chemistry package.

    assert ATOM_FDIM > 0  # Explanation: checks that atom feature dimension metadata is valid.
    assert BOND_FDIM > 0  # Explanation: checks that bond feature dimension metadata is valid.
    assert BondEditAction.__name__ == "BondEditAction"  # Explanation: checks the public action class resolves correctly.


def test_data_and_loss_imports():  # Explanation: verifies dataset and contrastive-loss exports are available.
    from kgcl_retro.data import RetroEditDataset, get_batch_graphs  # Explanation: imports packaged data helpers.
    from kgcl_retro.losses import ADNCE  # Explanation: imports the packaged ADNCE loss class.

    assert RetroEditDataset.__name__ == "RetroEditDataset"  # Explanation: checks the training dataset class resolves correctly.
    assert callable(get_batch_graphs)  # Explanation: checks the graph batching helper is callable.
    assert ADNCE.__name__ == "ADNCE"  # Explanation: checks the loss class resolves correctly.


def test_legacy_imports_still_work():  # Explanation: verifies old import paths remain compatible.
    from models import KGCL  # Explanation: imports the model through the legacy models package.
    from utils.rxn_graphs import MolGraph  # Explanation: imports the graph class through the legacy utils module.

    assert KGCL.__name__ == "KGCL"  # Explanation: checks the legacy model import resolves to the KGCL class.
    assert MolGraph.__name__ == "MolGraph"  # Explanation: checks the legacy graph import resolves to the MolGraph class.


def test_eval_roundtrip_compiles_without_syntax_warnings(tmp_path):
    import py_compile
    import warnings
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src" / "kgcl_retro" / "cli" / "eval_roundtrip.py"

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        py_compile.compile(str(source), cfile=str(tmp_path / "eval_roundtrip.pyc"), doraise=True)
