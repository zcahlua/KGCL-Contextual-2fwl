from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from kgcl_retro.chemistry.contextual_fg import INF_DISTANCE, MoleculeFGMetadata


@dataclass
class ContextualFGOutput:
    fg_embeddings: torch.Tensor
    atom_fg_context: torch.Tensor
    enhanced_atom_states: torch.Tensor
    fg_instance_scope: list[tuple[int, int]]
    diagnostics: dict


class ContextualFGEncoder(nn.Module):
    def __init__(
        self,
        atom_hidden_size: int,
        fg_hidden_size: int,
        fg_layers: int = 2,
        fg_use_boundary_pool: bool = True,
        fg_use_distance_bias: bool = True,
        fg_use_membership_bias: bool = True,
        kg_embedding_size: int = 256,
        max_fg_types: int = 512,
        max_distance_bucket: int = 8,
    ) -> None:
        super().__init__()
        self.atom_hidden_size = atom_hidden_size
        self.fg_hidden_size = fg_hidden_size
        self.fg_use_boundary_pool = fg_use_boundary_pool
        self.fg_use_distance_bias = fg_use_distance_bias
        self.fg_use_membership_bias = fg_use_membership_bias
        self.kg_embedding_size = kg_embedding_size
        self.max_distance_bucket = max_distance_bucket

        self.atom_project = nn.Linear(atom_hidden_size, fg_hidden_size)
        self.fg_type_embedding = nn.Embedding(max_fg_types, fg_hidden_size)
        self.context_mlp = nn.Sequential(
            nn.Linear(fg_hidden_size * 3, fg_hidden_size),
            nn.SELU(),
            *[
                layer
                for _ in range(max(0, fg_layers - 1))
                for layer in (nn.Linear(fg_hidden_size, fg_hidden_size), nn.SELU())
            ],
        )
        self.kg_project = nn.Linear(kg_embedding_size, fg_hidden_size)
        self.ctx_project = nn.Linear(fg_hidden_size, fg_hidden_size)
        self.gate = nn.Linear(kg_embedding_size + fg_hidden_size + 6, fg_hidden_size)
        self.null_fg = nn.Parameter(torch.zeros(fg_hidden_size))
        self.boundary_null = nn.Parameter(torch.zeros(fg_hidden_size))

        self.query = nn.Linear(atom_hidden_size, fg_hidden_size)
        self.key = nn.Linear(fg_hidden_size, fg_hidden_size)
        self.value = nn.Linear(fg_hidden_size, atom_hidden_size)
        self.output = nn.Linear(atom_hidden_size, atom_hidden_size)
        self.layer_norm = nn.LayerNorm(atom_hidden_size)
        self.distance_bias = nn.Embedding(max_distance_bucket + 2, 1)
        self.membership_in = nn.Parameter(torch.tensor(0.2))
        self.membership_out = nn.Parameter(torch.tensor(0.0))
        self.null_bias = nn.Parameter(torch.tensor(0.0))

    def _pool(self, atom_states: torch.Tensor, atom_indices: list[int], null_value: torch.Tensor) -> torch.Tensor:
        if not atom_indices:
            return null_value
        return atom_states[torch.tensor(atom_indices, dtype=torch.long, device=atom_states.device)].sum(dim=0)

    def _instance_embedding(
        self,
        atom_states: torch.Tensor,
        instance,
        atom_offset: int,
    ) -> torch.Tensor:
        if instance.is_null:
            return self.null_fg

        core_abs = [atom_offset + atom_idx for atom_idx in instance.core_atom_indices]
        boundary_abs = [
            atom_offset + atom_idx
            for atom_idx, is_boundary in zip(instance.context_atom_indices, instance.boundary_mask)
            if is_boundary
        ]
        projected_atoms = self.atom_project(atom_states)
        core_pool = self._pool(projected_atoms, core_abs, self.null_fg)
        boundary_pool = self._pool(projected_atoms, boundary_abs, self.boundary_null)
        type_idx = max(0, min(instance.fg_type_index, self.fg_type_embedding.num_embeddings - 1))
        type_emb = self.fg_type_embedding(
            torch.tensor(type_idx, dtype=torch.long, device=atom_states.device)
        )
        e_ctx = self.context_mlp(torch.cat([core_pool, boundary_pool, type_emb], dim=0))

        kg = torch.zeros(self.kg_embedding_size, device=atom_states.device, dtype=atom_states.dtype)
        if instance.kg_embedding is not None:
            raw_kg = torch.tensor(instance.kg_embedding, device=atom_states.device, dtype=atom_states.dtype).flatten()
            kg[: min(self.kg_embedding_size, raw_kg.numel())] = raw_kg[: self.kg_embedding_size]
        descriptors = torch.tensor(
            instance.chem_descriptors or [0.0] * 6,
            device=atom_states.device,
            dtype=atom_states.dtype,
        )
        kg_proj = self.kg_project(kg)
        ctx_proj = self.ctx_project(e_ctx)
        gate = torch.sigmoid(self.gate(torch.cat([kg, e_ctx, descriptors], dim=0)))
        return gate * ctx_proj + (1.0 - gate) * kg_proj

    def _distance_bucket(self, distance: int | str) -> int:
        if distance == "inf" or distance == INF_DISTANCE:
            return self.max_distance_bucket + 1
        return min(int(distance), self.max_distance_bucket)

    def forward(
        self,
        atom_states: torch.Tensor,
        fg_metadata: list[MoleculeFGMetadata],
        atom_scope: list[tuple[int, int]],
    ) -> ContextualFGOutput:
        fg_embeddings: list[torch.Tensor] = []
        fg_instance_scope: list[tuple[int, int]] = []
        atom_context = torch.zeros_like(atom_states)
        enhanced = atom_states.clone()

        for mol_idx, metadata in enumerate(fg_metadata):
            atom_offset, atom_count = atom_scope[mol_idx]
            start = len(fg_embeddings)
            mol_embeddings = [
                self._instance_embedding(atom_states, instance, atom_offset)
                for instance in metadata.instances
            ]
            if not mol_embeddings:
                mol_embeddings = [self.null_fg]
            fg_embeddings.extend(mol_embeddings)
            fg_instance_scope.append((start, len(mol_embeddings)))

            mol_fg = torch.stack(mol_embeddings, dim=0)
            q = self.query(atom_states[atom_offset: atom_offset + atom_count])
            k = self.key(mol_fg)
            v = self.value(mol_fg)
            scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(float(self.fg_hidden_size))

            for local_atom in range(atom_count):
                for fg_local, instance in enumerate(metadata.instances):
                    if instance.is_null:
                        scores[local_atom, fg_local] = scores[local_atom, fg_local] + self.null_bias
                        continue
                    if self.fg_use_membership_bias:
                        if local_atom in instance.core_atom_indices:
                            scores[local_atom, fg_local] = scores[local_atom, fg_local] + self.membership_in
                        else:
                            scores[local_atom, fg_local] = scores[local_atom, fg_local] + self.membership_out
                    if self.fg_use_distance_bias:
                        bucket = self._distance_bucket(instance.distance_to_core.get(local_atom, "inf"))
                        bias = self.distance_bias(
                            torch.tensor(bucket, dtype=torch.long, device=atom_states.device)
                        ).squeeze()
                        scores[local_atom, fg_local] = scores[local_atom, fg_local] + bias

            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, v)
            atom_context[atom_offset: atom_offset + atom_count] = context
            enhanced[atom_offset: atom_offset + atom_count] = self.layer_norm(
                atom_states[atom_offset: atom_offset + atom_count] + self.output(context)
            )

        stacked_fg = torch.stack(fg_embeddings, dim=0) if fg_embeddings else self.null_fg.unsqueeze(0)
        return ContextualFGOutput(
            fg_embeddings=stacked_fg,
            atom_fg_context=atom_context,
            enhanced_atom_states=enhanced,
            fg_instance_scope=fg_instance_scope,
            diagnostics={"num_fg_instances": int(stacked_fg.size(0))},
        )
