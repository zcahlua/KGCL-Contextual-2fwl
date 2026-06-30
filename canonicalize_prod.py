from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy wrapper can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # Explanation: lets this script run before kgcl-retro is installed.

from kgcl_retro.cli.canonicalize_products import canonicalize, canonicalize_prod, fix_charge, infer_correspondence, main, remap_rxn_smi  # Explanation: re-exports the original helper functions from the packaged CLI module.

__all__ = [  # Explanation: preserves the public API that existed in the original top-level script.
    "canonicalize",  # Explanation: exposes SMILES canonicalization.
    "canonicalize_prod",  # Explanation: exposes product canonicalization with atom-map assignment.
    "fix_charge",  # Explanation: exposes reactant/product charge repair.
    "infer_correspondence",  # Explanation: exposes atom-map correspondence inference.
    "main",  # Explanation: exposes the CLI entry point.
    "remap_rxn_smi",  # Explanation: exposes full reaction SMILES remapping.
]  # Explanation: closes the legacy public API list.


if __name__ == "__main__":  # Explanation: runs the compatibility entry point only when executed as a script.
    main()  # Explanation: invokes the packaged canonicalization command.
