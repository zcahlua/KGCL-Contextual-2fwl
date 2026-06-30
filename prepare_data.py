from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy wrapper can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # Explanation: lets this script run before kgcl-retro is installed.

from kgcl_retro.chemistry.apply import apply_edit_to_mol  # Explanation: preserves the old helper import path used by legacy code.
from kgcl_retro.cli.prepare_data import main, prepare_data, process_batch  # Explanation: re-exports packaged training-data preparation functions.

__all__ = ["apply_edit_to_mol", "main", "prepare_data", "process_batch"]  # Explanation: defines the legacy module API for external imports.


if __name__ == "__main__":  # Explanation: runs the compatibility entry point only when executed as a script.
    main()  # Explanation: invokes the packaged data-preparation command.
