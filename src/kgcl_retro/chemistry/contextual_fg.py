from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from kgcl_retro.chemistry.functional_groups import load_functional_group_resources


INF_DISTANCE = 10**9


@dataclass(frozen=True)
class FunctionalGroupInstance:
    fg_name: str
    fg_type_index: int
    core_atom_indices: tuple[int, ...]
    context_atom_indices: tuple[int, ...]
    distance_to_core: dict[int, int | str]
    core_mask: tuple[bool, ...]
    boundary_mask: tuple[bool, ...]
    kg_embedding: list[float] | Any | None
    chem_descriptors: list[float] | None
    is_null: bool = False


@dataclass(frozen=True)
class MoleculeFGMetadata:
    instances: list[FunctionalGroupInstance]
    atom_to_fg_core: list[list[int]]
    atom_to_fg_context: list[list[int]]
    atom_fg_distance: list[list[int]]
    has_null: bool
    num_atoms: int


def _embedding_set(use_rxn_class: bool) -> str:
    return "KGembedding_2" if use_rxn_class else "KGembedding"


def _adjacency(mol: Chem.Mol) -> list[list[int]]:
    graph = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        graph[begin].append(end)
        graph[end].append(begin)
    return graph


def _distances_from_core(mol: Chem.Mol, core: tuple[int, ...]) -> dict[int, int]:
    if not core:
        return {}
    graph = _adjacency(mol)
    distances: dict[int, int] = {atom_idx: 0 for atom_idx in core}
    queue: deque[int] = deque(core)
    while queue:
        atom_idx = queue.popleft()
        for neighbor in graph[atom_idx]:
            if neighbor not in distances:
                distances[neighbor] = distances[atom_idx] + 1
                queue.append(neighbor)
    return distances


def _chem_descriptors(mol: Chem.Mol, core: tuple[int, ...]) -> list[float]:
    if not core:
        return [0.0] * 6
    atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in core]
    ring_count = sum(1.0 for atom in atoms if atom.IsInRing())
    aromatic_count = sum(1.0 for atom in atoms if atom.GetIsAromatic())
    hetero_count = sum(1.0 for atom in atoms if atom.GetAtomicNum() not in (1, 6))
    formal_charge = float(sum(atom.GetFormalCharge() for atom in atoms))
    valence_sum = float(sum(atom.GetTotalValence() for atom in atoms))
    donor_acceptor_like = sum(
        1.0 for atom in atoms if atom.GetAtomicNum() in (7, 8, 15, 16) and atom.GetTotalValence() > 0
    )
    normalizer = float(max(len(core), 1))
    return [
        ring_count / normalizer,
        aromatic_count / normalizer,
        hetero_count / normalizer,
        formal_charge,
        valence_sum / normalizer,
        donor_acceptor_like / normalizer,
    ]


def _null_metadata(num_atoms: int) -> MoleculeFGMetadata:
    null_instance = FunctionalGroupInstance(
        fg_name="__null__",
        fg_type_index=-1,
        core_atom_indices=(),
        context_atom_indices=(),
        distance_to_core={},
        core_mask=(),
        boundary_mask=(),
        kg_embedding=None,
        chem_descriptors=[0.0] * 6,
        is_null=True,
    )
    return MoleculeFGMetadata(
        instances=[null_instance],
        atom_to_fg_core=[[] for _ in range(num_atoms)],
        atom_to_fg_context=[[] for _ in range(num_atoms)],
        atom_fg_distance=[[] for _ in range(num_atoms)],
        has_null=True,
        num_atoms=num_atoms,
    )


def match_functional_group_instances(
    mol: Chem.Mol,
    use_rxn_class: bool,
    radius: int,
    max_instances: int | None = None,
    include_null: bool = True,
) -> MoleculeFGMetadata:
    resources = load_functional_group_resources(_embedding_set(use_rxn_class))
    num_atoms = mol.GetNumAtoms()
    seen: set[tuple[str, tuple[int, ...]]] = set()
    instances: list[FunctionalGroupInstance] = []

    for fg_type_index, smarts in enumerate(resources.smarts):
        if smarts is None:
            continue
        fg_name = resources.smarts_to_name[smarts]
        matches = mol.GetSubstructMatches(smarts, uniquify=True)
        for match in matches:
            core = tuple(sorted(int(atom_idx) for atom_idx in match))
            dedupe_key = (fg_name, core)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            distances = _distances_from_core(mol, core)
            context = tuple(
                atom_idx
                for atom_idx in range(num_atoms)
                if distances.get(atom_idx, INF_DISTANCE) <= radius
            )
            distance_to_core: dict[int, int | str] = {
                atom_idx: distances.get(atom_idx, "inf") for atom_idx in context
            }
            core_set = set(core)
            core_mask = tuple(atom_idx in core_set for atom_idx in context)
            boundary_mask = tuple(atom_idx not in core_set for atom_idx in context)
            kg_embedding = resources.embeddings.get(fg_name)
            if hasattr(kg_embedding, "tolist"):
                kg_embedding = kg_embedding.tolist()

            instances.append(
                FunctionalGroupInstance(
                    fg_name=fg_name,
                    fg_type_index=fg_type_index,
                    core_atom_indices=core,
                    context_atom_indices=context,
                    distance_to_core=distance_to_core,
                    core_mask=core_mask,
                    boundary_mask=boundary_mask,
                    kg_embedding=kg_embedding,
                    chem_descriptors=_chem_descriptors(mol, core),
                )
            )
            if max_instances is not None and len(instances) >= max_instances:
                break
        if max_instances is not None and len(instances) >= max_instances:
            break

    if not instances and include_null:
        return _null_metadata(num_atoms)

    atom_to_fg_core = [[] for _ in range(num_atoms)]
    atom_to_fg_context = [[] for _ in range(num_atoms)]
    atom_fg_distance = [[] for _ in range(num_atoms)]
    for fg_idx, instance in enumerate(instances):
        for atom_idx in instance.core_atom_indices:
            atom_to_fg_core[atom_idx].append(fg_idx)
        for atom_idx in instance.context_atom_indices:
            atom_to_fg_context[atom_idx].append(fg_idx)
            distance = instance.distance_to_core.get(atom_idx, "inf")
            atom_fg_distance[atom_idx].append(INF_DISTANCE if distance == "inf" else int(distance))

    return MoleculeFGMetadata(
        instances=instances,
        atom_to_fg_core=atom_to_fg_core,
        atom_to_fg_context=atom_to_fg_context,
        atom_fg_distance=atom_fg_distance,
        has_null=any(instance.is_null for instance in instances),
        num_atoms=num_atoms,
    )
