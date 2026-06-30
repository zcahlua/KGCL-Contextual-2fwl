from __future__ import annotations

import torch
import torch.nn as nn

from kgcl_retro.chemistry.sparse_pair_builder import PAIR_RELATION_FEATURE_SIZE


class PairRelationEncoder(nn.Module):
    def __init__(self, pair_relation_size: int, feature_size: int = PAIR_RELATION_FEATURE_SIZE) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(3, pair_relation_size)
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_size, pair_relation_size),
            nn.SELU(),
            nn.Linear(pair_relation_size, pair_relation_size),
        )

    def forward(self, relation_features: torch.Tensor) -> torch.Tensor:
        if relation_features.numel() == 0:
            return torch.zeros((0, self.relation_embedding.embedding_dim), device=relation_features.device)
        if relation_features.dim() == 2:
            return self.feature_mlp(relation_features.float())
        return self.relation_embedding(relation_features.clamp(min=0, max=2))


class SparsePairLayerReference(nn.Module):
    def __init__(self, pair_hidden_size: int, pair_relation_size: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(pair_hidden_size * 3 + pair_relation_size * 3, pair_hidden_size),
            nn.SELU(),
            nn.Linear(pair_hidden_size, pair_hidden_size),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(pair_hidden_size * 2 + pair_relation_size, pair_hidden_size),
            nn.SELU(),
            nn.Linear(pair_hidden_size, pair_hidden_size),
        )
        self.layer_norm = nn.LayerNorm(pair_hidden_size)
        self.null_message = nn.Parameter(torch.zeros(pair_hidden_size))

    def forward(
        self,
        pair_states: torch.Tensor,
        pair_rel: torch.Tensor,
        pairs: torch.Tensor,
        bridge_index: torch.Tensor,
        bridge_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pairs.numel() == 0:
            return pair_states
        pair_lookup = {(int(i), int(j)): idx for idx, (i, j) in enumerate(pairs.tolist())}
        updated = []
        for pair_idx, (i, j) in enumerate(pairs.tolist()):
            messages = []
            for bridge_atom, enabled in zip(bridge_index[pair_idx].tolist(), bridge_mask[pair_idx].tolist()):
                if not enabled:
                    continue
                left_idx = pair_lookup.get((int(i), int(bridge_atom)))
                right_idx = pair_lookup.get((int(bridge_atom), int(j)))
                if left_idx is None or right_idx is None:
                    continue
                messages.append(
                    self.message_mlp(
                        torch.cat(
                            [
                                pair_states[left_idx],
                                pair_states[right_idx],
                                pair_states[pair_idx],
                                pair_rel[left_idx],
                                pair_rel[right_idx],
                                pair_rel[pair_idx],
                            ],
                            dim=0,
                        )
                    )
                )
            if messages:
                message = torch.stack(messages, dim=0).mean(dim=0)
            else:
                message = self.null_message
            delta = self.update_mlp(torch.cat([pair_states[pair_idx], message, pair_rel[pair_idx]], dim=0))
            updated.append(self.layer_norm(pair_states[pair_idx] + delta))
        return torch.stack(updated, dim=0)


class SparsePairLayerVectorized(nn.Module):
    def __init__(self, pair_hidden_size: int, pair_relation_size: int) -> None:
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(pair_hidden_size * 3 + pair_relation_size * 3, pair_hidden_size),
            nn.SELU(),
            nn.Linear(pair_hidden_size, pair_hidden_size),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(pair_hidden_size * 2 + pair_relation_size, pair_hidden_size),
            nn.SELU(),
            nn.Linear(pair_hidden_size, pair_hidden_size),
        )
        self.layer_norm = nn.LayerNorm(pair_hidden_size)
        self.null_message = nn.Parameter(torch.zeros(pair_hidden_size))

    @staticmethod
    def _lookup_pair_indices(pairs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pairs.numel() == 0 or targets.numel() == 0:
            empty = torch.zeros(targets.shape[:-1], dtype=torch.long, device=targets.device)
            return empty, torch.zeros(targets.shape[:-1], dtype=torch.bool, device=targets.device)
        max_value = torch.maximum(pairs.max(), targets.max()).to(torch.long)
        base = max_value + 1
        pair_keys = pairs[:, 0] * base + pairs[:, 1]
        target_keys = targets[..., 0] * base + targets[..., 1]
        sorted_keys, sorted_order = torch.sort(pair_keys)
        positions = torch.searchsorted(sorted_keys, target_keys.reshape(-1))
        in_range = positions < sorted_keys.numel()
        safe_positions = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
        matched = in_range & (sorted_keys[safe_positions] == target_keys.reshape(-1))
        indices = torch.zeros_like(safe_positions)
        if sorted_order.numel():
            indices = sorted_order[safe_positions]
        return indices.reshape(target_keys.shape), matched.reshape(target_keys.shape)

    def forward(
        self,
        pair_states: torch.Tensor,
        pair_rel: torch.Tensor,
        pairs: torch.Tensor,
        bridge_index: torch.Tensor,
        bridge_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pairs.numel() == 0:
            return pair_states
        if bridge_index.numel() == 0:
            messages = self.null_message.to(dtype=pair_states.dtype, device=pair_states.device).expand(pair_states.size(0), -1)
        else:
            left_targets = torch.stack(
                [
                    pairs[:, 0].unsqueeze(1).expand_as(bridge_index),
                    bridge_index,
                ],
                dim=-1,
            )
            right_targets = torch.stack(
                [
                    bridge_index,
                    pairs[:, 1].unsqueeze(1).expand_as(bridge_index),
                ],
                dim=-1,
            )
            left_idx, left_valid = self._lookup_pair_indices(pairs, left_targets)
            right_idx, right_valid = self._lookup_pair_indices(pairs, right_targets)
            valid = bridge_mask & left_valid & right_valid
            left_states = pair_states[left_idx]
            right_states = pair_states[right_idx]
            self_states = pair_states.unsqueeze(1).expand(-1, bridge_index.size(1), -1)
            left_rel = pair_rel[left_idx]
            right_rel = pair_rel[right_idx]
            self_rel = pair_rel.unsqueeze(1).expand(-1, bridge_index.size(1), -1)
            bridge_inputs = torch.cat([left_states, right_states, self_states, left_rel, right_rel, self_rel], dim=-1)
            bridge_messages = self.message_mlp(bridge_inputs.reshape(-1, bridge_inputs.size(-1))).reshape(
                pairs.size(0),
                bridge_index.size(1),
                pair_states.size(1),
            )
            valid_float = valid.unsqueeze(-1).to(dtype=bridge_messages.dtype)
            summed = (bridge_messages * valid_float).sum(dim=1)
            counts = valid_float.sum(dim=1)
            null_messages = self.null_message.to(dtype=pair_states.dtype, device=pair_states.device).expand(pair_states.size(0), -1)
            messages = torch.where(counts > 0, summed / counts.clamp(min=1.0), null_messages)

        delta = self.update_mlp(torch.cat([pair_states, messages, pair_rel], dim=1))
        return self.layer_norm(pair_states + delta)


class SparsePairLayer(SparsePairLayerVectorized):
    pass


class CandidateProposalHead(nn.Module):
    def __init__(self, atom_hidden_size: int, pair_relation_size: int, fg_pair_size: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(atom_hidden_size * 3 + pair_relation_size + fg_pair_size, atom_hidden_size),
            nn.SELU(),
            nn.Linear(atom_hidden_size, 1),
        )

    def forward(
        self,
        atom_states: torch.Tensor,
        unordered_pairs: torch.Tensor,
        pair_rel: torch.Tensor,
        fg_pair_context: torch.Tensor,
    ) -> torch.Tensor:
        if unordered_pairs.numel() == 0:
            return torch.zeros((0,), dtype=atom_states.dtype, device=atom_states.device)
        left = atom_states[unordered_pairs[:, 0]]
        right = atom_states[unordered_pairs[:, 1]]
        features = torch.cat([left + right, torch.abs(left - right), left * right, pair_rel, fg_pair_context], dim=1)
        return self.scorer(features).squeeze(-1)


class Sparse2FWLDecoder(nn.Module):
    def __init__(
        self,
        pair_hidden_size: int,
        pair_relation_size: int,
        pair_dec_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SparsePairLayer(pair_hidden_size, pair_relation_size) for _ in range(pair_dec_layers)]
        )

    def forward(
        self,
        pair_states: torch.Tensor,
        pair_rel: torch.Tensor,
        pairs: torch.Tensor,
        bridge_index: torch.Tensor,
        bridge_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = pair_states
        for layer in self.layers:
            output = layer(output, pair_rel, pairs, bridge_index, bridge_mask)
        return output
