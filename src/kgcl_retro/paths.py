from __future__ import annotations  # Explanation: enables modern type hints while preserving Python 3.10 compatibility.

from dataclasses import dataclass  # Explanation: imports dataclass for a small immutable path container.
from pathlib import Path  # Explanation: imports Path for robust filesystem path composition.


@dataclass(frozen=True)  # Explanation: keeps resolved project paths immutable during a CLI run.
class ProjectPaths:  # Explanation: collects repository-relative data and experiment paths for KGCL commands.
    root: Path  # Explanation: stores the root that contains data and experiment directories.

    @property  # Explanation: exposes data_dir as a computed read-only attribute.
    def data_dir(self) -> Path:  # Explanation: returns the root-level data directory.
        return self.root / "data"  # Explanation: builds the standard KGCL data path.

    @property  # Explanation: exposes experiments_dir as a computed read-only attribute.
    def experiments_dir(self) -> Path:  # Explanation: returns the root-level experiments directory.
        return self.root / "experiments"  # Explanation: builds the standard KGCL experiment-output path.

    def dataset_dir(self, dataset: str) -> Path:  # Explanation: resolves a named dataset inside the data directory.
        return self.data_dir / dataset  # Explanation: returns the directory for one USPTO dataset split collection.


def resolve_project_paths(root: str | Path = ".") -> ProjectPaths:  # Explanation: converts a CLI root argument into reusable path helpers.
    return ProjectPaths(root=Path(root).resolve())  # Explanation: normalizes the root path before downstream path construction.
