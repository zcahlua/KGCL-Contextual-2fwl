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
        bond_feature_size: int | None = None,
        fg_pool: str = "sum",
    ) -> None:
        super().__init__()
        self.atom_hidden_size = atom_hidden_size
        self.fg_hidden_size = fg_hidden_size
        if fg_pool not in {"sum", "mean", "max"}:
            raise ValueError("fg_pool must be one of: sum, mean, max")
        self.fg_pool = fg_pool
        self.fg_use_boundary_pool = fg_use_boundary_pool
        self.fg_use_distance_bias = fg_use_distance_bias
        self.fg_use_membership_bias = fg_use_membership_bias
        self.kg_embedding_size = kg_embedding_size
        self.max_distance_bucket = max_distance_bucket
        self.fg_layers = max(1, fg_layers)

        self.atom_project = nn.Linear(atom_hidden_size, fg_hidden_size)
        self.bond_project = nn.Linear(bond_feature_size or atom_hidden_size, fg_hidden_size)
        self.fg_type_embedding = nn.Embedding(max_fg_types, fg_hidden_size)
        self.core_embedding = nn.Embedding(2, fg_hidden_size)
        self.local_distance_embedding = nn.Embedding(max_distance_bucket + 2, fg_hidden_size)
        self.local_message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fg_hidden_size * 5 + 2, fg_hidden_size),
                    nn.SELU(),
                    nn.Linear(fg_hidden_size, fg_hidden_size),
                )
                for _ in range(self.fg_layers)
            ]
        )
        self.local_update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fg_hidden_size * 2, fg_hidden_size),
                    nn.SELU(),
                    nn.Linear(fg_hidden_size, fg_hidden_size),
                )
                for _ in range(self.fg_layers)
            ]
        )
        self.local_norms = nn.ModuleList([nn.LayerNorm(fg_hidden_size) for _ in range(self.fg_layers)])
        self.local_null_message = nn.Parameter(torch.zeros(fg_hidden_size))
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
        selected = atom_states[torch.tensor(atom_indices, dtype=torch.long, device=atom_states.device)]
        if self.fg_pool == "mean":
            return selected.mean(dim=0)
        if self.fg_pool == "max":
            return selected.max(dim=0).values
        return selected.sum(dim=0)

    def _context_edges(
        self,
        graph_tensors: tuple[torch.Tensor, ...] | None,
        context_abs: list[int],
    ) -> list[tuple[int, int, torch.Tensor]]:
        if graph_tensors is None or not context_abs:
            return []
        _f_atoms, f_bonds, _f_fgs, _atom_num, _n_mols, _a2b, b2a, b2revb, _undirected_b2a = graph_tensors
        context_lookup = {atom_abs: idx for idx, atom_abs in enumerate(context_abs)}
        edges: list[tuple[int, int, torch.Tensor]] = []
        for bond_idx in range(1, int(f_bonds.size(0))):
            src_abs = int(b2a[bond_idx].item())
            rev_idx = int(b2revb[bond_idx].item())
            if rev_idx <= 0 or rev_idx >= int(b2a.size(0)):
                continue
            dst_abs = int(b2a[rev_idx].item())
            if src_abs in context_lookup and dst_abs in context_lookup:
                edges.append((context_lookup[src_abs], context_lookup[dst_abs], f_bonds[bond_idx]))
        return edges

    def _run_local_mpnn(
        self,
        atom_states: torch.Tensor,
        instance,
        atom_offset: int,
        graph_tensors: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, list[int], list[bool], list[bool]]:
        context_abs = [atom_offset + atom_idx for atom_idx in instance.context_atom_indices]
        if not context_abs:
            return self.null_fg.unsqueeze(0), [], [], []
        core_set = {atom_offset + atom_idx for atom_idx in instance.core_atom_indices}
        context_local = [atom_abs - atom_offset for atom_abs in context_abs]
        core_mask = [atom_abs in core_set for atom_abs in context_abs]
        boundary_mask = [not flag for flag in core_mask]
        projected = self.atom_project(atom_states[torch.tensor(context_abs, dtype=torch.long, device=atom_states.device)])
        type_idx = max(0, min(instance.fg_type_index, self.fg_type_embedding.num_embeddings - 1))
        type_emb = self.fg_type_embedding(torch.tensor(type_idx, dtype=torch.long, device=atom_states.device))
        core_ids = torch.tensor([1 if flag else 0 for flag in core_mask], dtype=torch.long, device=atom_states.device)
        dist_ids = torch.tensor(
            [self._distance_bucket(instance.distance_to_core.get(local_atom, "inf")) for local_atom in context_local],
            dtype=torch.long,
            device=atom_states.device,
        )
        states = projected + self.core_embedding(core_ids) + self.local_distance_embedding(dist_ids) + type_emb
        edges = self._context_edges(graph_tensors, context_abs)
        dist_embeddings = self.local_distance_embedding(dist_ids)
        core_flags = core_ids.to(dtype=atom_states.dtype).unsqueeze(1)
        for layer_idx in range(self.fg_layers):
            messages = torch.zeros_like(states)
            counts = torch.zeros((states.size(0), 1), dtype=states.dtype, device=states.device)
            for src_idx, dst_idx, bond_feature in edges:
                bond_emb = self.bond_project(bond_feature.to(device=atom_states.device, dtype=atom_states.dtype))
                msg = self.local_message_mlps[layer_idx](
                    torch.cat(
                        [
                            states[dst_idx],
                            states[src_idx],
                            bond_emb,
                            dist_embeddings[dst_idx],
                            dist_embeddings[src_idx],
                            core_flags[dst_idx],
                            core_flags[src_idx],
                        ],
                        dim=0,
                    )
                )
                messages[dst_idx] = messages[dst_idx] + msg
                counts[dst_idx, 0] = counts[dst_idx, 0] + 1.0
            messages = torch.where(counts > 0.0, messages / counts.clamp(min=1.0), self.local_null_message)
            delta = self.local_update_mlps[layer_idx](torch.cat([states, messages], dim=1))
            states = self.local_norms[layer_idx](states + delta)
        return states, context_abs, core_mask, boundary_mask

    def _instance_embedding(
        self,
        atom_states: torch.Tensor,
        instance,
        atom_offset: int,
        graph_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if instance.is_null:
            return self.null_fg

        local_states, _context_abs, core_mask, boundary_mask = self._run_local_mpnn(
            atom_states, instance, atom_offset, graph_tensors
        )
        core_indices = [idx for idx, flag in enumerate(core_mask) if flag]
        boundary_indices = [idx for idx, flag in enumerate(boundary_mask) if flag]
        core_pool = self._pool(local_states, core_indices, self.null_fg)
        boundary_pool = self._pool(local_states, boundary_indices, self.boundary_null)
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
        graph_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> ContextualFGOutput:
        fg_embeddings: list[torch.Tensor] = []
        fg_instance_scope: list[tuple[int, int]] = []
        atom_context = torch.zeros_like(atom_states)
        enhanced = atom_states.clone()

        for mol_idx, metadata in enumerate(fg_metadata):
            atom_offset, atom_count = atom_scope[mol_idx]
            start = len(fg_embeddings)
            mol_embeddings = [
                self._instance_embedding(atom_states, instance, atom_offset, graph_tensors=graph_tensors)
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
