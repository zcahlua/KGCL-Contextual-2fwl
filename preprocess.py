from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy wrapper can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # Explanation: lets this script run before kgcl-retro is installed.

from kgcl_retro.cli.preprocess import main, preprocessing, check_edits  # Explanation: re-exports preprocessing helpers from the packaged CLI module.

__all__ = ["main", "preprocessing", "check_edits"]  # Explanation: defines the legacy module API for external imports.


if __name__ == "__main__":  # Explanation: runs the compatibility entry point only when executed as a script.
    main()  # Explanation: invokes the packaged preprocessing command.
