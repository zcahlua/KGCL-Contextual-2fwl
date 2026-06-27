from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from rdkit import Chem

from kgcl_retro.chemistry.contextual_fg import INF_DISTANCE, MoleculeFGMetadata


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
    gold_bond_pairs: torch.LongTensor = field(default_factory=lambda: torch.zeros((0, 2), dtype=torch.long))
    gold_atom_indices: torch.LongTensor = field(default_factory=lambda: torch.zeros((0,), dtype=torch.long))
    action_vector_lengths: list[int] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _to_tensor(pairs: Iterable[tuple[int, int]]) -> torch.LongTensor:
    ordered = sorted(set(pairs))
    if not ordered:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(ordered, dtype=torch.long)


def _distances(mol: Chem.Mol) -> list[list[int]]:
    n_atoms = mol.GetNumAtoms()
    distances = [[INF_DISTANCE] * n_atoms for _ in range(n_atoms)]
    for i in range(n_atoms):
        distances[i][i] = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        distances[i][j] = 1
        distances[j][i] = 1
    for k in range(n_atoms):
        for i in range(n_atoms):
            via_k = distances[i][k]
            if via_k == INF_DISTANCE:
                continue
            for j in range(n_atoms):
                candidate = via_k + distances[k][j]
                if candidate < distances[i][j]:
                    distances[i][j] = candidate
    return distances


def _relation_code(mol: Chem.Mol, i_abs: int, j_abs: int, atom_offset: int) -> int:
    i = i_abs - atom_offset
    j = j_abs - atom_offset
    if i == j:
        return 0
    return 1 if mol.GetBondBetweenAtoms(i, j) is not None else 2


def _fg_context_pairs(fg_metadata: MoleculeFGMetadata, atom_offset: int) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for instance in fg_metadata.instances:
        if instance.is_null:
            continue
        atoms = [atom_offset + atom_idx for atom_idx in instance.context_atom_indices]
        for i in atoms:
            for j in atoms:
                pairs.add((i, j))
    return pairs


def _score_pairs(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    atom_offset: int,
    pair_near_radius: int,
    max_score_pairs: int,
) -> set[tuple[int, int]]:
    n_atoms = mol.GetNumAtoms()
    distances = _distances(mol)
    pairs = {(atom_offset + i, atom_offset + i) for i in range(n_atoms)}
    for bond in mol.GetBonds():
        i = atom_offset + bond.GetBeginAtomIdx()
        j = atom_offset + bond.GetEndAtomIdx()
        pairs.add((i, j))
        pairs.add((j, i))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if distances[i][j] <= pair_near_radius:
                pairs.add((atom_offset + i, atom_offset + j))
    pairs.update(_fg_context_pairs(fg_metadata, atom_offset))
    pairs.update((j, i) for i, j in list(pairs))

    if len(pairs) <= max_score_pairs:
        return pairs
    diagonals = {(atom_offset + i, atom_offset + i) for i in range(n_atoms)}
    required = set(diagonals)
    for bond in mol.GetBonds():
        i = atom_offset + bond.GetBeginAtomIdx()
        j = atom_offset + bond.GetEndAtomIdx()
        required.add((i, j))
        required.add((j, i))
    optional = sorted(pairs - required)
    capped = set(required)
    capped.update(optional[: max(0, max_score_pairs - len(required))])
    capped.update((j, i) for i, j in list(capped))
    return capped


def _bridge_atoms(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    i_abs: int,
    j_abs: int,
    atom_offset: int,
    pair_bridge_radius: int,
    max_bridges: int,
) -> list[int]:
    distances = _distances(mol)
    i = i_abs - atom_offset
    j = j_abs - atom_offset
    bridge_atoms: set[int] = set()
    for u in range(mol.GetNumAtoms()):
        if distances[i][u] <= pair_bridge_radius and distances[u][j] <= pair_bridge_radius:
            bridge_atoms.add(atom_offset + u)
    for instance in fg_metadata.instances:
        if instance.is_null:
            continue
        context = {atom_offset + atom_idx for atom_idx in instance.context_atom_indices}
        if i_abs in context or j_abs in context:
            bridge_atoms.update(context)

    def rank(atom_abs: int) -> tuple[int, int, int]:
        u = atom_abs - atom_offset
        shortest_gap = abs((distances[i][u] + distances[u][j]) - distances[i][j])
        return (shortest_gap, distances[i][u] + distances[u][j], atom_abs)

    return sorted(bridge_atoms, key=rank)[:max_bridges]


def _carrier_pairs(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    score_pairs: set[tuple[int, int]],
    atom_offset: int,
    pair_bridge_radius: int,
    max_carrier_pairs: int,
    max_bridges: int,
) -> set[tuple[int, int]]:
    carrier = set(score_pairs)
    for i, j in sorted(score_pairs):
        for u in _bridge_atoms(mol, fg_metadata, i, j, atom_offset, pair_bridge_radius, max_bridges):
            carrier.update({(i, u), (u, i), (u, j), (j, u)})
    carrier.update((j, i) for i, j in list(carrier))
    if len(carrier) <= max_carrier_pairs:
        return carrier

    non_score = sorted(carrier - score_pairs)
    capped = set(score_pairs)
    for pair in non_score:
        if len(capped) >= max_carrier_pairs:
            break
        reverse = (pair[1], pair[0])
        if len(capped) + (0 if reverse in capped or reverse == pair else 1) + 1 <= max_carrier_pairs:
            capped.add(pair)
            capped.add(reverse)
    return capped


def _bridge_tensors(
    mol: Chem.Mol,
    fg_metadata: MoleculeFGMetadata,
    carrier_pairs: set[tuple[int, int]],
    atom_offset: int,
    pair_bridge_radius: int,
    max_bridges: int,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    rows: list[list[int]] = []
    masks: list[list[bool]] = []
    carrier = set(carrier_pairs)
    for i, j in sorted(carrier_pairs):
        candidates = _bridge_atoms(mol, fg_metadata, i, j, atom_offset, pair_bridge_radius, max_bridges * 2)
        closed = [u for u in candidates if (i, u) in carrier and (u, j) in carrier][:max_bridges]
        padded = closed + [0] * (max_bridges - len(closed))
        rows.append(padded)
        masks.append([True] * len(closed) + [False] * (max_bridges - len(closed)))
    if not rows:
        return torch.zeros((0, max_bridges), dtype=torch.long), torch.zeros((0, max_bridges), dtype=torch.bool)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def _unordered_candidates(dec_score_pairs: set[tuple[int, int]]) -> torch.LongTensor:
    unordered = sorted({tuple(sorted((i, j))) for i, j in dec_score_pairs if i != j})
    if not unordered:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(unordered, dtype=torch.long)


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
) -> SparsePairMetadata:
    enc_score = _score_pairs(mol, fg_metadata, atom_offset, pair_near_radius, pair_max_score_pairs_enc)
    dec_score = _score_pairs(mol, fg_metadata, atom_offset, pair_near_radius, pair_max_score_pairs_dec)
    enc_carrier = _carrier_pairs(
        mol, fg_metadata, enc_score, atom_offset, pair_bridge_radius, pair_max_carrier_pairs_enc, pair_max_bridges_enc
    )
    dec_carrier = _carrier_pairs(
        mol, fg_metadata, dec_score, atom_offset, pair_bridge_radius, pair_max_carrier_pairs_dec, pair_max_bridges_dec
    )
    enc_bridge_index, enc_bridge_mask = _bridge_tensors(
        mol, fg_metadata, enc_carrier, atom_offset, pair_bridge_radius, pair_max_bridges_enc
    )
    dec_bridge_index, dec_bridge_mask = _bridge_tensors(
        mol, fg_metadata, dec_carrier, atom_offset, pair_bridge_radius, pair_max_bridges_dec
    )
    enc_carrier_tensor = _to_tensor(enc_carrier)
    dec_carrier_tensor = _to_tensor(dec_carrier)
    relation_codes = torch.tensor(
        [_relation_code(mol, int(i), int(j), atom_offset) for i, j in enc_carrier_tensor.tolist()],
        dtype=torch.long,
    )
    dec_relation_codes = torch.tensor(
        [_relation_code(mol, int(i), int(j), atom_offset) for i, j in dec_carrier_tensor.tolist()],
        dtype=torch.long,
    )
    diagnostics = {
        "num_enc_score": len(enc_score),
        "num_enc_plus": len(enc_carrier),
        "num_dec_score": len(dec_score),
        "num_dec_plus": len(dec_carrier),
        "fraction_enc_nonempty_bridge": float(enc_bridge_mask.any(dim=1).float().mean().item()) if enc_bridge_mask.numel() else 0.0,
        "fraction_dec_nonempty_bridge": float(dec_bridge_mask.any(dim=1).float().mean().item()) if dec_bridge_mask.numel() else 0.0,
    }
    action_pairs = _unordered_candidates(dec_score)
    return SparsePairMetadata(
        enc_score_pairs=_to_tensor(enc_score),
        dec_score_pairs_base=_to_tensor(dec_score),
        enc_carrier_pairs=enc_carrier_tensor,
        dec_carrier_pairs_base=dec_carrier_tensor,
        enc_pair_scope=[(0, len(enc_carrier))],
        dec_pair_scope=[(0, len(dec_carrier))],
        enc_bridge_index=enc_bridge_index,
        enc_bridge_mask=enc_bridge_mask,
        dec_bridge_index_base=dec_bridge_index,
        dec_bridge_mask_base=dec_bridge_mask,
        pair_relation_codes=relation_codes,
        dec_pair_relation_codes=dec_relation_codes,
        unordered_dec_candidate_pairs=action_pairs,
        action_pair_scope=[(0, int(action_pairs.size(0)))],
        atom_scope=[(atom_offset, mol.GetNumAtoms())],
        action_vector_lengths=[],
        diagnostics=diagnostics,
    )


def merge_sparse_pair_metadata(items: list[SparsePairMetadata]) -> SparsePairMetadata:
    if not items:
        empty_pairs = torch.zeros((0, 2), dtype=torch.long)
        empty_bridge = torch.zeros((0, 0), dtype=torch.long)
        empty_mask = torch.zeros((0, 0), dtype=torch.bool)
        return SparsePairMetadata(
            empty_pairs, empty_pairs, empty_pairs, empty_pairs, [], [], empty_bridge, empty_mask,
            empty_bridge, empty_mask, torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long), empty_pairs, [], []
        )

    enc_pair_scope: list[tuple[int, int]] = []
    dec_pair_scope: list[tuple[int, int]] = []
    action_pair_scope: list[tuple[int, int]] = []
    atom_scope: list[tuple[int, int]] = []
    enc_cursor = dec_cursor = action_cursor = 0
    for item in items:
        enc_pair_scope.append((enc_cursor, int(item.enc_carrier_pairs.size(0))))
        dec_pair_scope.append((dec_cursor, int(item.dec_carrier_pairs_base.size(0))))
        action_pair_scope.append((action_cursor, int(item.unordered_dec_candidate_pairs.size(0))))
        atom_scope.extend(item.atom_scope)
        enc_cursor += int(item.enc_carrier_pairs.size(0))
        dec_cursor += int(item.dec_carrier_pairs_base.size(0))
        action_cursor += int(item.unordered_dec_candidate_pairs.size(0))

    diagnostics = {
        "avg_enc_score": sum(item.diagnostics.get("num_enc_score", 0) for item in items) / len(items),
        "avg_enc_plus": sum(item.diagnostics.get("num_enc_plus", 0) for item in items) / len(items),
        "avg_dec_score": sum(item.diagnostics.get("num_dec_score", 0) for item in items) / len(items),
        "avg_dec_plus": sum(item.diagnostics.get("num_dec_plus", 0) for item in items) / len(items),
    }
    return SparsePairMetadata(
        enc_score_pairs=torch.cat([item.enc_score_pairs for item in items], dim=0),
        dec_score_pairs_base=torch.cat([item.dec_score_pairs_base for item in items], dim=0),
        enc_carrier_pairs=torch.cat([item.enc_carrier_pairs for item in items], dim=0),
        dec_carrier_pairs_base=torch.cat([item.dec_carrier_pairs_base for item in items], dim=0),
        enc_pair_scope=enc_pair_scope,
        dec_pair_scope=dec_pair_scope,
        enc_bridge_index=torch.cat([item.enc_bridge_index for item in items], dim=0),
        enc_bridge_mask=torch.cat([item.enc_bridge_mask for item in items], dim=0),
        dec_bridge_index_base=torch.cat([item.dec_bridge_index_base for item in items], dim=0),
        dec_bridge_mask_base=torch.cat([item.dec_bridge_mask_base for item in items], dim=0),
        pair_relation_codes=torch.cat([item.pair_relation_codes for item in items], dim=0),
        dec_pair_relation_codes=torch.cat([item.dec_pair_relation_codes for item in items], dim=0),
        unordered_dec_candidate_pairs=torch.cat([item.unordered_dec_candidate_pairs for item in items], dim=0),
        action_pair_scope=action_pair_scope,
        atom_scope=atom_scope,
        diagnostics=diagnostics,
    )
