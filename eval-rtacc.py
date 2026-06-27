from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy wrapper can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # Explanation: lets this script run before kgcl-retro is installed.

from kgcl_retro.cli.eval_roundtrip import canonicalize, canonicalize_p, main, smi_tokenizer  # Explanation: re-exports packaged round-trip evaluation helpers.

__all__ = [  # Explanation: preserves the public API that existed in the original top-level round-trip script.
    "canonicalize",  # Explanation: exposes general SMILES canonicalization.
    "canonicalize_p",  # Explanation: exposes product canonicalization with atom maps.
    "main",  # Explanation: exposes the CLI entry point.
    "smi_tokenizer",  # Explanation: exposes the forward-model SMILES tokenizer.
]  # Explanation: closes the legacy public API list.


if __name__ == "__main__":  # Explanation: runs the compatibility entry point only when executed as a script.
    main()  # Explanation: invokes the packaged round-trip evaluation command.
