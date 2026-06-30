from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy module can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # Explanation: lets this module import the package before installation.

from kgcl_retro.models.utils import CSVLogger, creat_edits_feats, get_seq_edit_accuracy, index_select_ND, unbatch_feats  # Explanation: re-exports packaged model utilities.

__all__ = [  # Explanation: defines the legacy module API.
    "CSVLogger",  # Explanation: preserves access to the CSV training logger.
    "creat_edits_feats",  # Explanation: preserves access to atom feature padding for edit scoring.
    "get_seq_edit_accuracy",  # Explanation: preserves access to sequence-level edit accuracy.
    "index_select_ND",  # Explanation: preserves access to neighbor tensor indexing.
    "unbatch_feats",  # Explanation: preserves access to flattened atom feature restoration.
]  # Explanation: closes the legacy module API list.
