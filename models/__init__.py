from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy package can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # Explanation: lets legacy model imports work before kgcl-retro is installed.

from kgcl_retro.models import BeamSearch, KGCL  # Explanation: re-exports packaged model classes through the old models package.

__all__ = ["BeamSearch", "KGCL"]  # Explanation: defines the legacy models package API.
