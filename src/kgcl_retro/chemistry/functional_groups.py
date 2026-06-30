from __future__ import annotations  # Explanation: defers annotation evaluation so optional chemistry types do not need runtime imports.

from dataclasses import dataclass  # Explanation: imports dataclass for the immutable resource container.
from functools import lru_cache  # Explanation: imports lru_cache so asset files are loaded once per embedding set.
from importlib import resources  # Explanation: imports package-resource helpers for installed asset files.
import pickle  # Explanation: imports pickle to load the saved functional-group embedding dictionary.
from typing import Any  # Explanation: imports Any for RDKit and embedding objects whose concrete types come from external packages.


@dataclass(frozen=True)  # Explanation: makes the loaded functional-group resources immutable after construction.
class FunctionalGroupResources:  # Explanation: groups SMARTS patterns, names, and KG embeddings for graph construction.
    names: list[str]  # Explanation: stores functional-group names in the same order as the SMARTS patterns.
    smarts: list[Any]  # Explanation: stores compiled RDKit SMARTS molecules used for substructure matching.
    smarts_to_name: dict[Any, str]  # Explanation: maps each compiled SMARTS object back to its functional-group name.
    embeddings: dict[str, Any]  # Explanation: maps each functional-group name to its knowledge-graph embedding vector.


def _asset_root(embedding_set: str):  # Explanation: resolves the package asset directory for one embedding set.
    return resources.files("kgcl_retro").joinpath("assets", embedding_set)  # Explanation: returns a Traversable path inside the installed package.


@lru_cache(maxsize=2)  # Explanation: caches both KGembedding variants used by the paper's ablations.
def load_functional_group_resources(embedding_set: str) -> FunctionalGroupResources:  # Explanation: loads names, SMARTS patterns, and embeddings for KGCL graph features.
    from rdkit import Chem  # Explanation: imports RDKit lazily so package import still works before chemistry dependencies are installed.

    root = _asset_root(embedding_set)  # Explanation: locates the requested packaged embedding directory.
    funcgroup_text = root.joinpath("funcgroup.txt").read_text()  # Explanation: reads the functional-group SMARTS definition file.
    rows = [line.split() for line in funcgroup_text.strip().splitlines()]  # Explanation: tokenizes each functional-group row into name and SMARTS pattern.
    names = [row[0] for row in rows]  # Explanation: extracts the functional-group names used as embedding keys.
    smarts = [Chem.MolFromSmarts(row[1]) for row in rows]  # Explanation: compiles SMARTS strings into RDKit substructure query molecules.
    with root.joinpath("fg2emb.pkl").open("rb") as handle:  # Explanation: opens the packaged embedding pickle in binary mode.
        embeddings = pickle.load(handle)  # Explanation: deserializes the functional-group embedding dictionary.
    return FunctionalGroupResources(  # Explanation: returns one structured resource object to graph-building code.
        names=names,  # Explanation: preserves the functional-group names for diagnostics and tests.
        smarts=smarts,  # Explanation: passes compiled SMARTS patterns to substructure matching.
        smarts_to_name=dict(zip(smarts, names)),  # Explanation: builds the reverse lookup used after a SMARTS match.
        embeddings=embeddings,  # Explanation: passes the KG embedding vectors to graph feature fusion.
    )  # Explanation: closes construction of the functional-group resource container.
