from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy module can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # Explanation: lets this module import the package before installation.

from kgcl_retro.models.kgcl import KGCL  # Explanation: re-exports the packaged KGCL neural network class.

__all__ = ["KGCL"]  # Explanation: defines the legacy module API.
