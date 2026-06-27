from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy module can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # Explanation: lets this module import the package before installation.

from kgcl_retro.models.encoder import FeedForward, Global_Attention, MPNEncoder, MultiHeadAttention  # Explanation: re-exports packaged encoder layers.

__all__ = ["FeedForward", "Global_Attention", "MPNEncoder", "MultiHeadAttention"]  # Explanation: defines the legacy module API.
