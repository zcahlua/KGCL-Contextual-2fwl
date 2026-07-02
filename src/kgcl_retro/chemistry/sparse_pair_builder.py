from __future__ import annotations

import atexit
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
from rdkit import Chem

from kgcl_retro.chemistry.contextual_fg import INF_DISTANCE, MoleculeFGMetadata
from kgcl_retro.chemistry.features import BOND_FDIM, BOND_TYPES, get_bond_features


PAIR_RELATION_FEATURE_SIZE = 32

_PROFILE_LOCK = threading.Lock()
_PROFILE_STATS: dict[str, dict[str, float | int]] = {}


def _profile_enabled() -> bool:
    return os.environ.get("KGCL_PROFILE_SPARSE_PAIR", "").strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _profile_block(name: str):
    if not _profile_enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        with _PROFILE_LOCK:
            entry = _PROFILE_STATS.setdefault(name, {"calls": 0, "total": 0.0})
            entry["calls"] = int(entry["calls"]) + 1
            entry["total"] = float(entry["total"]) + elapsed


def reset_profile_stats() -> None:
    with _PROFILE_LOCK:
        _PROFILE_STATS.clear()


def get_profile_stats() -> dict[str, dict[str, float | int]]:
    if not _profile_enabled():
        return {}
    with _PROFILE_LOCK:
        return {name: dict(values) for name, values in _PROFILE_STATS.items()}


def _print_profile_stats() -> None:
    if not _profile_enabled():
        return
    stats = get_profile_stats()
    if not stats:
        return
    print("[sparse-pair-profile]")
    for name, values in sorted(stats.items(), key=lambda item: (-float(item[1]["total"]), item[0])):
        calls = int(values["calls"])
        total = float(values["total"])
        avg = total / max(calls, 1)
        print(f"{name:<30} calls={calls:<5d} total={total:.3f}s  avg={avg:.4f}s")


if not globals().get("_PROFILE_ATEXIT_REGISTERED", False):
    atexit.register(_print_profile_stats)
    _PROFILE_ATEXIT_REGISTERED = True


@dataclass
class ProposalPairMetadata:
    unordered_pairs: torch.LongTensor
    pair_relation_features: torch.FloatTensor
    pair_relation_codes: torch.LongTensor
    atom_scope: list[tuple[int, int]]
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SparsePairMetadata:
    enc_score_pairs: torch.LongTensor
    dec_score_pairs_base: torch.LongTensor
    enc_carrier_pairs: torch.LongTensor
    dec_carrier_pairs_base: torch.LongTensor
    enc_pair_scope: list[tuple[int, int]]
    dec_pair_scope: list[tuple[int, int]]
    enc_bridge_index: torch.LongTensor
    enc_bridge_mask: torch.BoolTensor
    dec_bridge_index_base: torch.LongTensor
    dec_bridge_mask_base: torch.BoolTensor
    pair_relation_codes: torch.LongTensor
    dec_pair_relation_codes: torch.LongTensor
    unordered_dec_candidate_pairs: torch.LongTensor
    action_pair_scope: list[tuple[int, int]]
    atom_scope: list[tuple[int, int]]
    pair_relation_features: torch.FloatTensor = field(
        default_factory=lambda: torch.zeros((0, PAIR_RELATION_FEATURE_SIZE), dtype=torch.float32)
    )
    dec_pair_relation_features: torch.FloatTensor = field(
        default_factory=lambda: torch.zeros((0, PAIR_RELATION_FEATURE_SIZE), dtype=torch.float32)
    )
    proposal_universe_pairs: torch.LongTensor = field(default_factory=lambda: torch.zeros((0, 2), dtype=torch.long))
    proposal_pair_relation_features: torch.FloatTensor = field(
        default_factory=lambda: torch.zeros((0, PAIR_RELATION_FEATURE_SIZE), dtype=torch.float32)
    )
    proposal_pair_relation_codes: torch.LongTensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    proposal_pair_scope: list[tuple[int, int]] = field(default_factory=list)
    gold_bond_pairs: torch.LongTensor = field(default_factory=lambda: torch.zeros((0, 2), dtype=torch.long))
    gold_atom_indices: torch.LongTensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    action_vector_lengths: list[int] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PairBuilderCache:
    mol: Chem.Mol
    atom_offset: int
    num_atoms: int
    distances: list[list[int]]
    fg_contexts_abs: list[frozenset[int]]
    atom_to_fg_context_abs: list[frozenset[int]]
    atom_to_fg_core_abs: list[frozenset[int]]
    same_ring_matrix: list[list[bool]]
    same_aromatic_system_matrix: list[list[bool]]
    bond_feature_map: dict[tuple[int, int], list[float]]
    bond_type_map: dict[tuple[int, int], list[float]]
    bond_exists: set[tuple[int, int]]
    fg_any_count_map: dict[tuple[int, int], float]
    fg_co_count_map: dict[tuple[int, int], float]


def _to_tensor(pairs: Iterable[tuple[int, int]]) -> torch.LongTensor:
    ordered = sorted(set(pairs))
    if not ordered:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(ordered, dtype=torch.long)


def _feature_tensor(features: list[list[float]]) -> torch.FloatTensor:
    if not features:
        return torch.zeros((0, PAIR_RELATION_FEATURE_SIZE), dtype=torch.float32)
    return torch.tensor(features, dtype=torch.float32)


def _distances(mol: Chem.Mol) -> list[list[int]]:
    with _profile_block("_distances"):
        n_atoms = mol.GetNumAtoms()
        adjacency = [[] for _ in range(n_atoms)]
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adjacency[i].append(j)
            adjacency[j].append(i)

        distances: list[list[int]] = []
        for source in range(n_atoms):
            row = [INF_DISTANCE] * n_atoms
            row[source] = 0
            queue: deque[int] = deque([source])
            while queue:
                atom_idx = queue.popleft()
                for neighbor in adjacency[atom_idx]:
                    if row[neighbor] == INF_DISTANCE:
                        row[neighbor] = row[atom_idx] + 1
                        queue.append(neighbor)
            distances.append(row)
        return distances


def _relation_code(
    mol: Chem.Mol,
    i_abs: int,
    j_abs: int,
    atom_offset: int,
    cache: PairBuilderCache | None = None,
) -> int:
    i = i_abs - atom_offset
    j = j_abs - atom_offset
    if i == j:
        return 0
    if cache is not None:
        return 1 if (i_abs, j_abs) in cache.bond_exists else 2
    return 1 if mol.GetBondBetweenAtoms(i, j) is not None else 2


def _fg_context_sets(fg_metadata: MoleculeFGMetadata, atom_offset: int) -> list[set[int]]:
    contexts = []
    for instance in fg_metadata.instances:
        if instance.is_null:
            continue
        contexts.append({atom_offset + atom_idx for atom_idx in instance.context_atom_indices})
    return contexts


def _fg_context_pairs(fg_metadata: MoleculeFGMetadata, atom_offset: int) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for context in _fg_context_sets(fg_metadata, atom_offset):
        for i in context:
            for j in context:
                pairs.add((i, j))
    return pairs


def _same_ring(mol: Chem.Mol, i: int, j: int) -> bool:
    with _profile_block("_same_ring"):
        if i == j:
            return True
        return any(i in ring and j in ring for ring in mol.GetRingInfo().AtomRings())


def _same_aromatic_system(mol: Chem.Mol, i: int, j: int) -> bool:
    with _profile_block("_same_aromatic_system"):
        if i == j:
            return mol.GetAtomWithIdx(i).GetIsAromatic()
        return any(
            i in ring
            and j in ring
            and all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
            for ring in mol.GetRingInfo().AtomRings()
        )


def _fg_counts(fg_contexts: list[set[int]], i_abs: int, j_abs: int) -> tuple[float, float]:
    with _profile_block("_fg_counts"):
        any_count = 0
        co_count = 0
        for context in fg_contexts:
            has_i = i_abs in context
            has_j = j_abs in context
            if has_i or has_j:
                any_count += 1
            if has_i and has_j:
                co_count += 1
        normalizer = float(max(len(fg_contexts), 1))
        return any_count / normalizer, co_count / normalizer


def _distance_bucket(distance: int) -> int:
    if distance == INF_DISTANCE:
        return 4
    return min(distance, 4)


def _bond_type_one_hot(bond: Chem.Bond | None) -> list[float]:
    if bond is None:
        return [0.0] * len(BOND_TYPES)
    return [float(bond.GetBondType() == bond_type) for bond_type in BOND_TYPES]


def _fg_context_pairs_from_cache(cache: PairBuilderCache) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for context in cache.fg_contexts_abs:
        for i in context:
            for j in context:
                pairs.add((i, j))
    return pairs


def _pair_in_atom_scope(pair: tuple[int, int], atom_offset: int, num_atoms: int) -> bool:
    start = atom_offset
    end = atom_offset + num_atoms
    return start <= pair[0] < end and start <= pair[1] < end


def _ring_matrix(mol: Chem.Mol) -> list[list[bool]]:
    n_atoms = mol.GetNumAtoms()
    matrix = [[i == j for j in range(n_atoms)] for i in range(n_atoms)]
    for ring in mol.GetRingInfo().AtomRings():
        ring_set = set(ring)
        for i in ring_set:
            for j in ring_set:
                matrix[i][j] = True
    return matrix


def _aromatic_system_matrix(mol: Chem.Mol) -> list[list[bool]]:
    n_atoms = mol.GetNumAtoms()
    matrix = [[False] * n_atoms for _ in range(n_atoms)]
    for i in range(n_atoms):
        matrix[i][i] = mol.GetAtomWithIdx(i).GetIsAromatic()
    for ring in mol.GetRingInfo().AtomRings():
        if not all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            continue
        ring_set = set(ring)
        for i in ring_set:
            for j in ring_set:
                matrix[i][j] = True
    return matrix


def build_pair_builder_cache(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    atom_offset: int = 1,
) -> PairBuilderCache:
    distances = _distances(mol)
    n_atoms = mol.GetNumAtoms()
    fg_contexts_abs = [
        frozenset(atom_offset + atom_idx for atom_idx in instance.context_atom_indices)
        for instance in fg_metadata.instances
        if not instance.is_null
    ]
    atom_to_fg_context_abs = [
        frozenset(fg_metadata.atom_to_fg_context[atom_idx])
        if atom_idx < len(fg_metadata.atom_to_fg_context)
        else frozenset()
        for atom_idx in range(n_atoms)
    ]
    atom_to_fg_core_abs = [
        frozenset(fg_metadata.atom_to_fg_core[atom_idx])
        if atom_idx < len(fg_metadata.atom_to_fg_core)
        else frozenset()
        for atom_idx in range(n_atoms)
    ]
    same_ring_matrix = _ring_matrix(mol)
    same_aromatic_system_matrix = _aromatic_system_matrix(mol)
    bond_feature_map: dict[tuple[int, int], list[float]] = {}
    bond_type_map: dict[tuple[int, int], list[float]] = {}
    bond_exists: set[tuple[int, int]] = set()
    for bond in mol.GetBonds():
        i_abs = atom_offset + bond.GetBeginAtomIdx()
        j_abs = atom_offset + bond.GetEndAtomIdx()
        raw_bond = (get_bond_features(bond) + [0.0] * BOND_FDIM)[:BOND_FDIM]
        bond_type = _bond_type_one_hot(bond)
        for pair in ((i_abs, j_abs), (j_abs, i_abs)):
            bond_exists.add(pair)
            bond_feature_map[pair] = list(raw_bond)
            bond_type_map[pair] = list(bond_type)

    fg_any_count_map: dict[tuple[int, int], float] = {}
    fg_co_count_map: dict[tuple[int, int], float] = {}
    normalizer = float(max(len(fg_contexts_abs), 1))
    for i in range(n_atoms):
        i_abs = atom_offset + i
        for j in range(n_atoms):
            j_abs = atom_offset + j
            any_count = 0
            co_count = 0
            for context in fg_contexts_abs:
                has_i = i_abs in context
                has_j = j_abs in context
                if has_i or has_j:
                    any_count += 1
                if has_i and has_j:
                    co_count += 1
            fg_any_count_map[(i_abs, j_abs)] = any_count / normalizer
            fg_co_count_map[(i_abs, j_abs)] = co_count / normalizer

    return PairBuilderCache(
        mol=mol,
        atom_offset=atom_offset,
        num_atoms=n_atoms,
        distances=distances,
        fg_contexts_abs=fg_contexts_abs,
        atom_to_fg_context_abs=atom_to_fg_context_abs,
        atom_to_fg_core_abs=atom_to_fg_core_abs,
        same_ring_matrix=same_ring_matrix,
        same_aromatic_system_matrix=same_aromatic_system_matrix,
        bond_feature_map=bond_feature_map,
        bond_type_map=bond_type_map,
        bond_exists=bond_exists,
        fg_any_count_map=fg_any_count_map,
        fg_co_count_map=fg_co_count_map,
    )


def _relation_feature(
    mol: Chem.Mol,
    fg_contexts_or_metadata: list[set[int]] | MoleculeFGMetadata,
    distances: list[list[int]],
    i_abs: int,
    j_abs: int,
    atom_offset: int,
    cache: PairBuilderCache | None = None,
) -> list[float]:
    with _profile_block("_relation_feature"):
        i = i_abs - atom_offset
        j = j_abs - atom_offset
        relation_code = _relation_code(mol, i_abs, j_abs, atom_offset, cache=cache)
        distance = distances[i][j]
        bucket = _distance_bucket(distance)
        distance_one_hot = [float(bucket == idx) for idx in range(5)]
        if cache is None:
            if isinstance(fg_contexts_or_metadata, MoleculeFGMetadata):
                fg_contexts = _fg_context_sets(fg_contexts_or_metadata, atom_offset)
            else:
                fg_contexts = fg_contexts_or_metadata
            fg_any, fg_co = _fg_counts(fg_contexts, i_abs, j_abs)
            bond = None if i == j else mol.GetBondBetweenAtoms(i, j)
            raw_bond = get_bond_features(bond) if bond is not None else [0.0] * BOND_FDIM
            raw_bond = (raw_bond + [0.0] * BOND_FDIM)[:BOND_FDIM]
            bond_type = _bond_type_one_hot(bond)
            same_ring = _same_ring(mol, i, j)
            same_aromatic = _same_aromatic_system(mol, i, j)
        else:
            fg_any = cache.fg_any_count_map.get((i_abs, j_abs), 0.0)
            fg_co = cache.fg_co_count_map.get((i_abs, j_abs), 0.0)
            raw_bond = cache.bond_feature_map.get((i_abs, j_abs), [0.0] * BOND_FDIM)
            bond_type = cache.bond_type_map.get((i_abs, j_abs), [0.0] * len(BOND_TYPES))
            same_ring = cache.same_ring_matrix[i][j]
            same_aromatic = cache.same_aromatic_system_matrix[i][j]
        features = [
            float(relation_code == 0),
            float(relation_code == 1),
            float(relation_code == 2),
            float(distance == INF_DISTANCE),
            *distance_one_hot,
            float(same_ring),
            float(fg_co > 0.0),
            float(same_aromatic),
            *bond_type,
            *raw_bond,
            fg_any,
            fg_co,
            float(i_abs < j_abs),
            1.0,
        ]
        if len(features) > PAIR_RELATION_FEATURE_SIZE:
            return features[:PAIR_RELATION_FEATURE_SIZE]
        return features + [0.0] * (PAIR_RELATION_FEATURE_SIZE - len(features))


def _relation_tensors(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    pairs: torch.LongTensor,
    atom_offset: int,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    with _profile_block("_relation_tensors"):
        if cache is None:
            if distances is None:
                distances = _distances(mol)
            fg_contexts: list[set[int]] | MoleculeFGMetadata = _fg_context_sets(fg_metadata, atom_offset)
        else:
            distances = cache.distances
            fg_contexts = fg_metadata
        codes = []
        features = []
        for i_abs, j_abs in pairs.tolist():
            codes.append(_relation_code(mol, int(i_abs), int(j_abs), atom_offset, cache=cache))
            features.append(
                _relation_feature(
                    mol,
                    fg_contexts,
                    distances,
                    int(i_abs),
                    int(j_abs),
                    atom_offset,
                    cache=cache,
                )
            )
        return torch.tensor(codes, dtype=torch.long), _feature_tensor(features)


def _score_pairs(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    atom_offset: int,
    pair_near_radius: int,
    max_score_pairs: int,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> set[tuple[int, int]]:
    with _profile_block("_score_pairs"):
        score_pairs, _ = _build_encoder_score_pairs(
            mol,
            fg_metadata,
            atom_offset,
            pair_near_radius,
            max_score_pairs,
            distances=distances,
            cache=cache,
        )
        return score_pairs


def _encoder_score_pair_sources(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    atom_offset: int,
    pair_near_radius: int,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> tuple[dict[str, set[tuple[int, int]]], set[tuple[int, int]], list[list[int]]]:
    if cache is None:
        cache = build_pair_builder_cache(mol, fg_metadata, atom_offset)
    distances = cache.distances
    n_atoms = cache.num_atoms

    diag = {(atom_offset + i, atom_offset + i) for i in range(n_atoms)}
    bond = {
        pair
        for pair in cache.bond_exists
        if _pair_in_atom_scope(pair, atom_offset, n_atoms)
    }

    near: set[tuple[int, int]] = set()
    for i in range(n_atoms):
        i_abs = atom_offset + i
        for j in range(n_atoms):
            if i == j:
                continue
            distance = distances[i][j]
            if distance != INF_DISTANCE and distance <= pair_near_radius:
                j_abs = atom_offset + j
                near.add((i_abs, j_abs))
                near.add((j_abs, i_abs))

    fg = {
        (i, j)
        for i, j in _fg_context_pairs_from_cache(cache)
        if i != j and _pair_in_atom_scope((i, j), atom_offset, n_atoms)
    }

    sources = {
        "diag": _with_reversals(diag),
        "bond": _with_reversals(bond),
        "near": _with_reversals(near),
        "fg": _with_reversals(fg),
    }
    uncapped = _with_reversals(set().union(*sources.values()))
    return sources, uncapped, distances


def _encoder_score_pair_rank(
    pair: tuple[int, int],
    sources: dict[str, set[tuple[int, int]]],
    distances: list[list[int]],
    atom_offset: int,
) -> tuple[int, int, int, tuple[int, int]]:
    if pair in sources["diag"] or pair in sources["bond"]:
        return (0, 0, 0, pair)
    in_near = pair in sources["near"]
    in_fg = pair in sources["fg"]
    i = pair[0] - atom_offset
    j = pair[1] - atom_offset
    distance = distances[i][j] if 0 <= i < len(distances) and 0 <= j < len(distances[i]) else INF_DISTANCE
    if in_near:
        return (1, distance, 0 if in_fg else 1, pair)
    if in_fg:
        return (2, INF_DISTANCE, 0, pair)
    return (3, INF_DISTANCE, 1, pair)


def _encoder_score_diagnostics(
    capped: set[tuple[int, int]],
    sources: dict[str, set[tuple[int, int]]],
    uncapped: set[tuple[int, int]],
) -> dict[str, int]:
    return {
        "num_enc_score_diag": len(capped & sources["diag"]),
        "num_enc_score_bond": len(capped & sources["bond"]),
        "num_enc_score_near": len(capped & sources["near"]),
        "num_enc_score_fg": len(capped & sources["fg"]),
        "num_enc_score_uncapped": len(uncapped),
        "num_enc_score_capped": len(capped),
        "num_enc_score_dropped_by_cap": len(uncapped - capped),
    }


def _build_encoder_score_pairs(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    atom_offset: int,
    pair_near_radius: int,
    max_score_pairs: int,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> tuple[set[tuple[int, int]], dict[str, int]]:
    sources, uncapped, distances = _encoder_score_pair_sources(
        mol,
        fg_metadata,
        atom_offset,
        pair_near_radius,
        distances=distances,
        cache=cache,
    )
    required = sources["diag"] | sources["bond"]
    capped = _cap_reversal_closed(
        uncapped,
        max_score_pairs,
        required,
        key_fn=lambda pair: _encoder_score_pair_rank(pair, sources, distances, atom_offset),
    )
    return capped, _encoder_score_diagnostics(capped, sources, uncapped)


def _required_pairs(mol: Chem.Mol, atom_offset: int) -> set[tuple[int, int]]:
    required = {(atom_offset + i, atom_offset + i) for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i = atom_offset + bond.GetBeginAtomIdx()
        j = atom_offset + bond.GetEndAtomIdx()
        required.add((i, j))
        required.add((j, i))
    return required


def _with_reversals(pairs: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    closed = set(pairs)
    closed.update((j, i) for i, j in list(closed))
    return closed


def _cap_reversal_closed(
    pairs: set[tuple[int, int]],
    max_pairs: int,
    required: set[tuple[int, int]],
    key_fn: Callable[[tuple[int, int]], tuple] | None = None,
) -> set[tuple[int, int]]:
    pairs = _with_reversals(pairs)
    required = _with_reversals(required)
    if len(pairs) <= max_pairs:
        return pairs
    capped = set(required)
    optional = sorted(pairs - required) if key_fn is None else sorted(pairs - required, key=key_fn)
    for pair in optional:
        reverse = (pair[1], pair[0])
        addition = {pair, reverse}
        if len(capped | addition) <= max_pairs:
            capped.update(addition)
    return capped


def _normalize_unordered_pairs(pairs: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs if int(i) != int(j)}


def _bridge_atoms(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    i_abs: int,
    j_abs: int,
    atom_offset: int,
    pair_bridge_radius: int,
    max_bridges: int,
    proposal_pairs: set[tuple[int, int]] | None = None,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> list[int]:
    with _profile_block("_bridge_atoms"):
        if cache is not None:
            distances = cache.distances
            fg_contexts = cache.fg_contexts_abs
            num_atoms = cache.num_atoms
        else:
            if distances is None:
                distances = _distances(mol)
            fg_contexts = [
                frozenset(atom_offset + atom_idx for atom_idx in instance.context_atom_indices)
                for instance in fg_metadata.instances
                if not instance.is_null
            ]
            num_atoms = mol.GetNumAtoms()
        i = i_abs - atom_offset
        j = j_abs - atom_offset
        bridge_atoms: set[int] = set()
        for u in range(num_atoms):
            if distances[i][u] <= pair_bridge_radius and distances[u][j] <= pair_bridge_radius:
                bridge_atoms.add(atom_offset + u)
        for context in fg_contexts:
            if i_abs in context or j_abs in context:
                bridge_atoms.update(context)
        proposal_pairs = proposal_pairs or set()
        for u_abs in range(atom_offset, atom_offset + num_atoms):
            if tuple(sorted((i_abs, u_abs))) in proposal_pairs or tuple(sorted((u_abs, j_abs))) in proposal_pairs:
                bridge_atoms.add(u_abs)

        def rank(atom_abs: int) -> tuple[int, int, int]:
            u = atom_abs - atom_offset
            shortest_gap = abs((distances[i][u] + distances[u][j]) - distances[i][j])
            proposal_bonus = 0 if tuple(sorted((i_abs, atom_abs))) in proposal_pairs else 1
            return (proposal_bonus, shortest_gap, distances[i][u] + distances[u][j], atom_abs)

        return sorted(bridge_atoms, key=rank)[:max_bridges]


def _carrier_pairs(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    score_pairs: set[tuple[int, int]],
    atom_offset: int,
    pair_bridge_radius: int,
    max_carrier_pairs: int,
    max_bridges: int,
    proposal_pairs: set[tuple[int, int]] | None = None,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> set[tuple[int, int]]:
    with _profile_block("_carrier_pairs"):
        if cache is not None:
            distances = cache.distances
        elif distances is None:
            distances = _distances(mol)
        carrier = set(score_pairs)
        for i, j in sorted(score_pairs):
            for u in _bridge_atoms(
                mol,
                fg_metadata,
                i,
                j,
                atom_offset,
                pair_bridge_radius,
                max_bridges,
                proposal_pairs=proposal_pairs,
                distances=distances,
                cache=cache,
            ):
                carrier.update({(i, u), (u, i), (u, j), (j, u)})
        return _cap_reversal_closed(carrier, max_carrier_pairs, score_pairs)


def _bridge_tensors(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    carrier_pairs: set[tuple[int, int]],
    atom_offset: int,
    pair_bridge_radius: int,
    max_bridges: int,
    proposal_pairs: set[tuple[int, int]] | None = None,
    blocked_bridge_pairs: set[tuple[int, int]] | None = None,
    protected_pairs: set[tuple[int, int]] | None = None,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    with _profile_block("_bridge_tensors"):
        if cache is not None:
            distances = cache.distances
        elif distances is None:
            distances = _distances(mol)
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        carrier = set(carrier_pairs)
        blocked = _with_reversals(blocked_bridge_pairs or set())
        protected = _with_reversals(protected_pairs or set())
        for i, j in sorted(carrier_pairs):
            candidates = _bridge_atoms(
                mol,
                fg_metadata,
                i,
                j,
                atom_offset,
                pair_bridge_radius,
                max_bridges * 2,
                proposal_pairs=proposal_pairs,
                distances=distances,
                cache=cache,
            )
            closed = []
            for u in candidates:
                if (i, u) not in carrier or (u, j) not in carrier:
                    continue
                if (i, j) not in protected and ((i, u) in blocked or (u, j) in blocked):
                    continue
                closed.append(u)
                if len(closed) == max_bridges:
                    break
            padded = closed + [0] * (max_bridges - len(closed))
            rows.append(padded)
            masks.append([True] * len(closed) + [False] * (max_bridges - len(closed)))
        if not rows:
            return torch.zeros((0, max_bridges), dtype=torch.long), torch.zeros((0, max_bridges), dtype=torch.bool)
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def _unordered_candidates(dec_score_pairs: set[tuple[int, int]]) -> torch.LongTensor:
    return _to_tensor(_normalize_unordered_pairs(dec_score_pairs))


def _avg_bridge_count(mask: torch.BoolTensor) -> float:
    return float(mask.float().sum(dim=1).mean().item()) if mask.numel() else 0.0


def build_encoder_pair_metadata(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    atom_offset: int = 1,
    pair_near_radius: int = 2,
    pair_bridge_radius: int = 2,
    pair_max_score_pairs_enc: int = 512,
    pair_max_carrier_pairs_enc: int = 1024,
    pair_max_bridges_enc: int = 8,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> SparsePairMetadata:
    with _profile_block("build_encoder_pair_metadata"):
        if cache is None:
            cache = build_pair_builder_cache(mol, fg_metadata, atom_offset)
        distances = cache.distances
        enc_score, enc_score_diagnostics = _build_encoder_score_pairs(
            mol,
            fg_metadata,
            atom_offset,
            pair_near_radius,
            pair_max_score_pairs_enc,
            distances=distances,
            cache=cache,
        )
        enc_carrier = _carrier_pairs(
            mol,
            fg_metadata,
            enc_score,
            atom_offset,
            pair_bridge_radius,
            pair_max_carrier_pairs_enc,
            pair_max_bridges_enc,
            distances=distances,
            cache=cache,
        )
        enc_bridge_index, enc_bridge_mask = _bridge_tensors(
            mol,
            fg_metadata,
            enc_carrier,
            atom_offset,
            pair_bridge_radius,
            pair_max_bridges_enc,
            distances=distances,
            cache=cache,
        )
        enc_carrier_tensor = _to_tensor(enc_carrier)
        relation_codes, relation_features = _relation_tensors(
            mol,
            fg_metadata,
            enc_carrier_tensor,
            atom_offset,
            distances=distances,
            cache=cache,
        )
        empty_pairs = torch.zeros((0, 2), dtype=torch.long)
        empty_bridge = torch.zeros((0, pair_max_bridges_enc), dtype=torch.long)
        empty_mask = torch.zeros((0, pair_max_bridges_enc), dtype=torch.bool)
        diagnostics = {
            "num_atoms": mol.GetNumAtoms(),
            "num_fg_instances": len([instance for instance in fg_metadata.instances if not instance.is_null]),
            "num_enc_score": len(enc_score),
            "num_enc_plus": len(enc_carrier),
            "avg_K_ij_enc": _avg_bridge_count(enc_bridge_mask),
            "fraction_enc_nonempty_bridge": float(enc_bridge_mask.any(dim=1).float().mean().item()) if enc_bridge_mask.numel() else 0.0,
            **enc_score_diagnostics,
        }
        return SparsePairMetadata(
            enc_score_pairs=_to_tensor(enc_score),
            dec_score_pairs_base=empty_pairs,
            enc_carrier_pairs=enc_carrier_tensor,
            dec_carrier_pairs_base=empty_pairs,
            enc_pair_scope=[(0, len(enc_carrier))],
            dec_pair_scope=[(0, 0)],
            enc_bridge_index=enc_bridge_index,
            enc_bridge_mask=enc_bridge_mask,
            dec_bridge_index_base=empty_bridge,
            dec_bridge_mask_base=empty_mask,
            pair_relation_codes=relation_codes,
            dec_pair_relation_codes=torch.zeros((0,), dtype=torch.long),
            unordered_dec_candidate_pairs=empty_pairs,
            action_pair_scope=[(0, 0)],
            atom_scope=[(atom_offset, mol.GetNumAtoms())],
            pair_relation_features=relation_features,
            diagnostics=diagnostics,
        )


def build_proposal_universe(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    atom_offset: int = 1,
    pair_near_radius: int = 2,
    pair_max_pairs: int = 2048,
    tiny_mol_all_pairs_cutoff: int = 16,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> ProposalPairMetadata:
    with _profile_block("build_proposal_universe"):
        if cache is None:
            cache = build_pair_builder_cache(mol, fg_metadata, atom_offset)
        n_atoms = mol.GetNumAtoms()
        distances = cache.distances if distances is None else distances
        unordered: set[tuple[int, int]] = set()
        for bond in mol.GetBonds():
            unordered.add(tuple(sorted((atom_offset + bond.GetBeginAtomIdx(), atom_offset + bond.GetEndAtomIdx()))))
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if distances[i][j] <= pair_near_radius:
                    unordered.add((atom_offset + i, atom_offset + j))
        for i, j in _fg_context_pairs_from_cache(cache):
            if i != j:
                unordered.add(tuple(sorted((i, j))))
        if n_atoms <= tiny_mol_all_pairs_cutoff:
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    unordered.add((atom_offset + i, atom_offset + j))
        ordered = sorted(unordered)[:pair_max_pairs]
        pair_tensor = _to_tensor(ordered)
        relation_codes, relation_features = _relation_tensors(
            mol,
            fg_metadata,
            pair_tensor,
            atom_offset,
            distances=distances,
            cache=cache,
        )
        return ProposalPairMetadata(
            unordered_pairs=pair_tensor,
            pair_relation_features=relation_features,
            pair_relation_codes=relation_codes,
            atom_scope=[(atom_offset, n_atoms)],
            diagnostics={"num_proposal_universe": len(ordered)},
        )


def build_decoder_pair_metadata(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    enc_score_pairs: Iterable[tuple[int, int]],
    proposal_topk_pairs: Iterable[tuple[int, int]],
    gold_bond_pairs: Iterable[tuple[int, int]] | None = None,
    training: bool = False,
    atom_offset: int = 1,
    pair_near_radius: int = 2,
    pair_bridge_radius: int = 2,
    pair_max_score_pairs_dec: int = 1024,
    pair_max_carrier_pairs_dec: int = 2048,
    pair_max_bridges_dec: int = 8,
    distances: list[list[int]] | None = None,
    cache: PairBuilderCache | None = None,
) -> SparsePairMetadata:
    with _profile_block("build_decoder_pair_metadata"):
        del pair_near_radius
        if cache is None:
            cache = build_pair_builder_cache(mol, fg_metadata, atom_offset)
        distances = cache.distances if distances is None else distances
        enc_score = set(enc_score_pairs)
        proposal_unordered = _normalize_unordered_pairs(proposal_topk_pairs)
        proposal_directed = _with_reversals(proposal_unordered)
        gold_unordered = _normalize_unordered_pairs(gold_bond_pairs or [])
        inference_candidates = _normalize_unordered_pairs(enc_score) | proposal_unordered
        rescued = gold_unordered - inference_candidates if training else set()
        score_pairs = set(enc_score)
        score_pairs.update(proposal_directed)
        if training:
            score_pairs.update(_with_reversals(gold_unordered))
        required = _required_pairs(mol, atom_offset)
        required.update(enc_score)
        if training:
            required.update(_with_reversals(gold_unordered))
        score_pairs = _cap_reversal_closed(score_pairs, pair_max_score_pairs_dec, required)
        carrier = _carrier_pairs(
            mol,
            fg_metadata,
            score_pairs,
            atom_offset,
            pair_bridge_radius,
            pair_max_carrier_pairs_dec,
            pair_max_bridges_dec,
            proposal_pairs=proposal_unordered,
            distances=distances,
            cache=cache,
        )
        gold_rescued_directed = _with_reversals(rescued)
        bridge_index, bridge_mask = _bridge_tensors(
            mol,
            fg_metadata,
            carrier,
            atom_offset,
            pair_bridge_radius,
            pair_max_bridges_dec,
            proposal_pairs=proposal_unordered,
            blocked_bridge_pairs=gold_rescued_directed,
            protected_pairs=gold_rescued_directed,
            distances=distances,
            cache=cache,
        )
        carrier_tensor = _to_tensor(carrier)
        relation_codes, relation_features = _relation_tensors(
            mol,
            fg_metadata,
            carrier_tensor,
            atom_offset,
            distances=distances,
            cache=cache,
        )
        action_pairs = _unordered_candidates(score_pairs)
        absent_in_inference = gold_unordered - inference_candidates
        diagnostics = {
            "num_dec_score": len(score_pairs),
            "num_dec_plus": len(carrier),
            "avg_K_ij_dec": _avg_bridge_count(bridge_mask),
            "fraction_dec_nonempty_bridge": float(bridge_mask.any(dim=1).float().mean().item()) if bridge_mask.numel() else 0.0,
            "gold_pairs_absent_from_inference": len(absent_in_inference),
            "gold_pairs_rescued_by_teacher_forcing": len(rescued),
            "gold_only_bridge_pairs_blocked": len(gold_rescued_directed),
        }
        empty_pairs = torch.zeros((0, 2), dtype=torch.long)
        return SparsePairMetadata(
            enc_score_pairs=empty_pairs,
            dec_score_pairs_base=_to_tensor(score_pairs),
            enc_carrier_pairs=empty_pairs,
            dec_carrier_pairs_base=carrier_tensor,
            enc_pair_scope=[(0, 0)],
            dec_pair_scope=[(0, len(carrier))],
            enc_bridge_index=torch.zeros((0, pair_max_bridges_dec), dtype=torch.long),
            enc_bridge_mask=torch.zeros((0, pair_max_bridges_dec), dtype=torch.bool),
            dec_bridge_index_base=bridge_index,
            dec_bridge_mask_base=bridge_mask,
            pair_relation_codes=torch.zeros((0,), dtype=torch.long),
            dec_pair_relation_codes=relation_codes,
            unordered_dec_candidate_pairs=action_pairs,
            action_pair_scope=[(0, int(action_pairs.size(0)))],
            atom_scope=[(atom_offset, mol.GetNumAtoms())],
            dec_pair_relation_features=relation_features,
            diagnostics=diagnostics,
        )


def build_sparse_pair_metadata(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    atom_offset: int = 1,
    pair_near_radius: int = 2,
    pair_bridge_radius: int = 2,
    pair_max_score_pairs_enc: int = 512,
    pair_max_score_pairs_dec: int = 1024,
    pair_max_carrier_pairs_enc: int = 1024,
    pair_max_carrier_pairs_dec: int = 2048,
    pair_max_bridges_enc: int = 8,
    pair_max_bridges_dec: int = 8,
    pair_topk: int = 64,
    build_decoder_base: bool = False,
) -> SparsePairMetadata:
    with _profile_block("build_sparse_pair_metadata"):
        cache = build_pair_builder_cache(mol, fg_metadata, atom_offset)
        distances = cache.distances
        encoder = build_encoder_pair_metadata(
            mol,
            fg_metadata,
            atom_offset=atom_offset,
            pair_near_radius=pair_near_radius,
            pair_bridge_radius=pair_bridge_radius,
            pair_max_score_pairs_enc=pair_max_score_pairs_enc,
            pair_max_carrier_pairs_enc=pair_max_carrier_pairs_enc,
            pair_max_bridges_enc=pair_max_bridges_enc,
            distances=distances,
            cache=cache,
        )
        proposal = build_proposal_universe(
            mol,
            fg_metadata,
            atom_offset=atom_offset,
            pair_near_radius=pair_near_radius,
            pair_max_pairs=max(pair_topk, pair_max_score_pairs_dec),
            distances=distances,
            cache=cache,
        )
        encoder.proposal_universe_pairs = proposal.unordered_pairs
        encoder.proposal_pair_relation_features = proposal.pair_relation_features
        encoder.proposal_pair_relation_codes = proposal.pair_relation_codes
        encoder.proposal_pair_scope = [(0, int(proposal.unordered_pairs.size(0)))]
        encoder.diagnostics = {
            **encoder.diagnostics,
            **proposal.diagnostics,
            "build_decoder_base": build_decoder_base,
        }
        if not build_decoder_base:
            encoder.diagnostics.update(
                {
                    "num_dec_score": 0,
                    "num_dec_plus": 0,
                    "avg_K_ij_dec": 0.0,
                    "fraction_dec_nonempty_bridge": 0.0,
                }
            )
            return encoder

        decoder = build_decoder_pair_metadata(
            mol,
            fg_metadata,
            enc_score_pairs=set(map(tuple, encoder.enc_score_pairs.tolist())),
            proposal_topk_pairs=set(),
            atom_offset=atom_offset,
            pair_bridge_radius=pair_bridge_radius,
            pair_max_score_pairs_dec=pair_max_score_pairs_dec,
            pair_max_carrier_pairs_dec=pair_max_carrier_pairs_dec,
            pair_max_bridges_dec=pair_max_bridges_dec,
            training=False,
            distances=distances,
            cache=cache,
        )
        encoder.dec_score_pairs_base = decoder.dec_score_pairs_base
        encoder.dec_carrier_pairs_base = decoder.dec_carrier_pairs_base
        encoder.dec_pair_scope = decoder.dec_pair_scope
        encoder.dec_bridge_index_base = decoder.dec_bridge_index_base
        encoder.dec_bridge_mask_base = decoder.dec_bridge_mask_base
        encoder.dec_pair_relation_codes = decoder.dec_pair_relation_codes
        encoder.dec_pair_relation_features = decoder.dec_pair_relation_features
        encoder.unordered_dec_candidate_pairs = decoder.unordered_dec_candidate_pairs
        encoder.action_pair_scope = decoder.action_pair_scope
        encoder.diagnostics = {**encoder.diagnostics, **decoder.diagnostics}
        return encoder


def merge_decoder_pair_metadata(items: list[SparsePairMetadata]) -> dict:
    dec_pair_scope: list[tuple[int, int]] = []
    action_pair_scope: list[tuple[int, int]] = []
    dec_cursor = action_cursor = 0
    for item in items:
        dec_count = int(item.dec_carrier_pairs_base.size(0))
        action_count = int(item.unordered_dec_candidate_pairs.size(0))
        dec_pair_scope.append((dec_cursor, dec_count))
        action_pair_scope.append((action_cursor, action_count))
        dec_cursor += dec_count
        action_cursor += action_count
    empty_pairs = torch.zeros((0, 2), dtype=torch.long)
    empty_features = torch.zeros((0, PAIR_RELATION_FEATURE_SIZE), dtype=torch.float32)
    diagnostics = {}
    if items:
        diagnostics = {
            "avg_dec_score": sum(item.diagnostics.get("num_dec_score", 0) for item in items) / len(items),
            "avg_dec_plus": sum(item.diagnostics.get("num_dec_plus", 0) for item in items) / len(items),
            "avg_K_ij_dec": sum(item.diagnostics.get("avg_K_ij_dec", 0.0) for item in items) / len(items),
            "fraction_dec_nonempty_bridge": sum(
                item.diagnostics.get("fraction_dec_nonempty_bridge", 0.0) for item in items
            )
            / len(items),
            "gold_pairs_absent_from_inference": sum(
                item.diagnostics.get("gold_pairs_absent_from_inference", 0) for item in items
            ),
            "gold_pairs_rescued_by_teacher_forcing": sum(
                item.diagnostics.get("gold_pairs_rescued_by_teacher_forcing", 0) for item in items
            ),
        }
    return {
        "dec_score_pairs_base": torch.cat([item.dec_score_pairs_base for item in items], dim=0) if items else empty_pairs,
        "dec_carrier_pairs_base": torch.cat([item.dec_carrier_pairs_base for item in items], dim=0)
        if items
        else empty_pairs,
        "dec_pair_scope": dec_pair_scope,
        "dec_bridge_index_base": torch.cat([item.dec_bridge_index_base for item in items], dim=0)
        if items
        else torch.zeros((0, 0), dtype=torch.long),
        "dec_bridge_mask_base": torch.cat([item.dec_bridge_mask_base for item in items], dim=0)
        if items
        else torch.zeros((0, 0), dtype=torch.bool),
        "dec_pair_relation_codes": torch.cat([item.dec_pair_relation_codes for item in items], dim=0)
        if items
        else torch.zeros((0,), dtype=torch.long),
        "dec_pair_relation_features": torch.cat([item.dec_pair_relation_features for item in items], dim=0)
        if items
        else empty_features,
        "unordered_dec_candidate_pairs": torch.cat([item.unordered_dec_candidate_pairs for item in items], dim=0)
        if items
        else empty_pairs,
        "action_pair_scope": action_pair_scope,
        "diagnostics": diagnostics,
    }


def merge_sparse_pair_metadata(items: list[SparsePairMetadata]) -> SparsePairMetadata:
    if not items:
        empty_pairs = torch.zeros((0, 2), dtype=torch.long)
        empty_bridge = torch.zeros((0, 0), dtype=torch.long)
        empty_mask = torch.zeros((0, 0), dtype=torch.bool)
        empty_features = torch.zeros((0, PAIR_RELATION_FEATURE_SIZE), dtype=torch.float32)
        return SparsePairMetadata(
            enc_score_pairs=empty_pairs,
            dec_score_pairs_base=empty_pairs,
            enc_carrier_pairs=empty_pairs,
            dec_carrier_pairs_base=empty_pairs,
            enc_pair_scope=[],
            dec_pair_scope=[],
            enc_bridge_index=empty_bridge,
            enc_bridge_mask=empty_mask,
            dec_bridge_index_base=empty_bridge,
            dec_bridge_mask_base=empty_mask,
            pair_relation_codes=torch.zeros((0,), dtype=torch.long),
            dec_pair_relation_codes=torch.zeros((0,), dtype=torch.long),
            unordered_dec_candidate_pairs=empty_pairs,
            action_pair_scope=[],
            atom_scope=[],
            pair_relation_features=empty_features,
            dec_pair_relation_features=empty_features,
            proposal_pair_relation_features=empty_features,
            diagnostics={},
        )

    enc_pair_scope: list[tuple[int, int]] = []
    atom_scope: list[tuple[int, int]] = []
    proposal_pair_scope: list[tuple[int, int]] = []
    enc_cursor = proposal_cursor = 0
    for item in items:
        enc_pair_scope.append((enc_cursor, int(item.enc_carrier_pairs.size(0))))
        proposal_pair_scope.append((proposal_cursor, int(item.proposal_universe_pairs.size(0))))
        atom_scope.extend(item.atom_scope)
        enc_cursor += int(item.enc_carrier_pairs.size(0))
        proposal_cursor += int(item.proposal_universe_pairs.size(0))

    decoder = merge_decoder_pair_metadata(items)
    diagnostics = {
        "avg_enc_score": sum(item.diagnostics.get("num_enc_score", 0) for item in items) / len(items),
        "avg_enc_plus": sum(item.diagnostics.get("num_enc_plus", 0) for item in items) / len(items),
        "avg_K_ij_enc": sum(item.diagnostics.get("avg_K_ij_enc", 0.0) for item in items) / len(items),
        "fraction_enc_nonempty_bridge": sum(
            item.diagnostics.get("fraction_enc_nonempty_bridge", 0.0) for item in items
        )
        / len(items),
        "avg_dec_score": decoder["diagnostics"].get("avg_dec_score", 0.0),
        "avg_dec_plus": decoder["diagnostics"].get("avg_dec_plus", 0.0),
        "avg_K_ij_dec": decoder["diagnostics"].get("avg_K_ij_dec", 0.0),
        "fraction_dec_nonempty_bridge": decoder["diagnostics"].get("fraction_dec_nonempty_bridge", 0.0),
        "avg_proposal_universe": sum(item.diagnostics.get("num_proposal_universe", 0) for item in items) / len(items),
    }
    return SparsePairMetadata(
        enc_score_pairs=torch.cat([item.enc_score_pairs for item in items], dim=0),
        dec_score_pairs_base=decoder["dec_score_pairs_base"],
        enc_carrier_pairs=torch.cat([item.enc_carrier_pairs for item in items], dim=0),
        dec_carrier_pairs_base=decoder["dec_carrier_pairs_base"],
        enc_pair_scope=enc_pair_scope,
        dec_pair_scope=decoder["dec_pair_scope"],
        enc_bridge_index=torch.cat([item.enc_bridge_index for item in items], dim=0),
        enc_bridge_mask=torch.cat([item.enc_bridge_mask for item in items], dim=0),
        dec_bridge_index_base=decoder["dec_bridge_index_base"],
        dec_bridge_mask_base=decoder["dec_bridge_mask_base"],
        pair_relation_codes=torch.cat([item.pair_relation_codes for item in items], dim=0),
        dec_pair_relation_codes=decoder["dec_pair_relation_codes"],
        unordered_dec_candidate_pairs=decoder["unordered_dec_candidate_pairs"],
        action_pair_scope=decoder["action_pair_scope"],
        atom_scope=atom_scope,
        pair_relation_features=torch.cat([item.pair_relation_features for item in items], dim=0),
        dec_pair_relation_features=decoder["dec_pair_relation_features"],
        proposal_universe_pairs=torch.cat([item.proposal_universe_pairs for item in items], dim=0),
        proposal_pair_relation_features=torch.cat([item.proposal_pair_relation_features for item in items], dim=0),
        proposal_pair_relation_codes=torch.cat([item.proposal_pair_relation_codes for item in items], dim=0),
        proposal_pair_scope=proposal_pair_scope,
        diagnostics=diagnostics,
    )
