from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy package can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # Explanation: lets legacy utility imports work before kgcl-retro is installed.
