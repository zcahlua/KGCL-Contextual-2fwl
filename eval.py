from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy wrapper can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # Explanation: lets this script run before kgcl-retro is installed.

from kgcl_retro.cli.eval_50k import canonicalize_smiles, canonicalize_smiles_clear_map, main  # Explanation: re-exports packaged USPTO-50K evaluation helpers.

__all__ = [  # Explanation: preserves the public API that existed in the original top-level evaluation script.
    "canonicalize_smiles",  # Explanation: exposes SMILES canonicalization for predictions.
    "canonicalize_smiles_clear_map",  # Explanation: exposes canonicalization with largest-fragment extraction.
    "main",  # Explanation: exposes the CLI entry point.
]  # Explanation: closes the legacy public API list.


if __name__ == "__main__":  # Explanation: runs the compatibility entry point only when executed as a script.
    main()  # Explanation: invokes the packaged USPTO-50K evaluation command.
