from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kgcl_retro.chemistry.sparse_pair_builder import (
    build_decoder_pair_metadata,
    merge_decoder_pair_metadata,
)
from kgcl_retro.data.collate import ContextualEditTarget, GraphBatch
from kgcl_retro.models.contextual_fg import ContextualFGEncoder
from kgcl_retro.models.sparse_pair import (
    CandidateProposalHead,
    PairRelationEncoder,
    Sparse2FWLDecoder,
    SparsePairLayer,
)
from kgcl_retro.models.utils import index_select_ND


class ContextualFGKGCL2FWL(nn.Module):
    def __init__(self, config: dict[str, Any], atom_vocab, bond_vocab) -> None:
        super().__init__()
        self.config = config
        self.atom_vocab = atom_vocab
        self.bond_vocab = bond_vocab
        self.atom_outdim = len(atom_vocab)
        self.bond_outdim = len(bond_vocab)
        self.hidden_size = config["mpn_size"]
        self.pair_hidden_size = config.get("pair_hidden_size", self.hidden_size)
        self.pair_relation_size = config.get("pair_relation_size", max(16, self.hidden_size // 2))

        self.atom_feature_project = nn.Linear(config["n_atom_feat"], self.hidden_size)
        self.edge_input = nn.Linear(config["n_bond_feat"], self.hidden_size, bias=False)
        self.edge_gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.edge_dropout = nn.Dropout(p=config["dropout_mpn"])
        self.atom_update = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.SELU(),
        )
        self.fg_encoder = ContextualFGEncoder(
            atom_hidden_size=self.hidden_size,
            fg_hidden_size=config.get("fg_hidden_size", self.hidden_size),
            fg_layers=config.get("fg_layers", 2),
            fg_use_boundary_pool=config.get("fg_use_boundary_pool", True),
            fg_use_distance_bias=config.get("fg_use_distance_bias", True),
            fg_use_membership_bias=config.get("fg_use_membership_bias", True),
            kg_embedding_size=config.get("kg_embedding_size", 256),
            bond_feature_size=config["n_bond_feat"],
        )
        self.relation_encoder = PairRelationEncoder(self.pair_relation_size)
        pair_init_dim = self.hidden_size * 4 + self.pair_relation_size + 1
        self.pair_init = nn.Sequential(
            nn.Linear(pair_init_dim, self.pair_hidden_size),
            nn.SELU(),
            nn.Linear(self.pair_hidden_size, self.pair_hidden_size),
        )
        self.reuse_pair = nn.Linear(self.pair_hidden_size, self.pair_hidden_size)
        self.encoder_pair_layers = nn.ModuleList(
            [
                SparsePairLayer(self.pair_hidden_size, self.pair_relation_size)
                for _ in range(config.get("pair_enc_layers", 1))
            ]
        )
        self.decoder = Sparse2FWLDecoder(
            self.pair_hidden_size,
            self.pair_relation_size,
            config.get("pair_dec_layers", 1),
        )
        self.pair_to_atom = nn.Linear(self.pair_hidden_size + self.pair_relation_size, self.hidden_size)
        self.feedback_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.feedback_norm = nn.LayerNorm(self.hidden_size)
        self.proposal_head = CandidateProposalHead(
            self.hidden_size,
            self.pair_relation_size,
            self.hidden_size * 2,
        )

        self.atom_linear = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + self.pair_hidden_size, config["mlp_size"]),
            nn.SELU(),
            nn.Dropout(p=config["dropout_mlp"]),
            nn.Linear(config["mlp_size"], self.atom_outdim),
        )
        self.bond_rep = nn.Sequential(
            nn.Linear(self.pair_hidden_size * 3, self.pair_hidden_size),
            nn.SELU(),
            nn.Linear(self.pair_hidden_size, self.pair_hidden_size),
        )
        self.bond_linear = nn.Sequential(
            nn.Linear(self.pair_hidden_size + self.pair_relation_size + self.hidden_size * 2, config["mlp_size"]),
            nn.SELU(),
            nn.Dropout(p=config["dropout_mlp"]),
            nn.Linear(config["mlp_size"], self.bond_outdim),
        )
        self.graph_linear = nn.Sequential(
            nn.Linear(self.hidden_size + self.pair_hidden_size, config["mlp_size"]),
            nn.SELU(),
            nn.Dropout(p=config["dropout_mlp"]),
            nn.Linear(config["mlp_size"], 1),
        )
        self.last_diagnostics: dict[str, Any] = {}
        self.last_proposal_loss: torch.Tensor | None = None

    def _init_pair_states(
        self,
        atom_states: torch.Tensor,
        atom_fg_context: torch.Tensor,
        pair_rel: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        if pairs.numel() == 0:
            return torch.zeros((0, self.pair_hidden_size), dtype=atom_states.dtype, device=atom_states.device)
        left = atom_states[pairs[:, 0]]
        right = atom_states[pairs[:, 1]]
        left_fg = atom_fg_context[pairs[:, 0]]
        right_fg = atom_fg_context[pairs[:, 1]]
        diag = (pairs[:, 0] == pairs[:, 1]).to(dtype=atom_states.dtype, device=atom_states.device).unsqueeze(1)
        return self.pair_init(torch.cat([left, right, pair_rel, left_fg, right_fg, diag], dim=1))

    def _apply_pair_feedback(
        self,
        atom_states: torch.Tensor,
        pair_states: torch.Tensor,
        pair_rel: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        if pairs.numel() == 0:
            return atom_states
        feedback = torch.zeros_like(atom_states)
        counts = torch.zeros((atom_states.size(0), 1), dtype=atom_states.dtype, device=atom_states.device)
        projected = self.pair_to_atom(torch.cat([pair_states, pair_rel], dim=1))
        for pair_idx, (i, _j) in enumerate(pairs.tolist()):
            feedback[int(i)] = feedback[int(i)] + projected[pair_idx]
            counts[int(i), 0] = counts[int(i), 0] + 1.0
        feedback = feedback / counts.clamp(min=1.0)
        gate = torch.sigmoid(self.feedback_gate(torch.cat([atom_states, feedback], dim=1)))
        return self.feedback_norm(atom_states + gate * feedback)

    def _encode_contextual_graph(self, graph_batch: GraphBatch):
        base_tensors = graph_batch.base_tensors
        f_atoms, f_bonds, _f_fgs, _atom_num, _n_mols, a2b, b2a, b2revb, _undirected_b2a = base_tensors
        atom_scope, _bond_scope = graph_batch.scopes
        sparse = graph_batch.sparse_metadata
        assert sparse is not None

        preliminary_atoms = self.atom_feature_project(f_atoms)
        fg_out = self.fg_encoder(
            preliminary_atoms,
            graph_batch.fg_metadata or [],
            atom_scope,
            graph_tensors=base_tensors,
        )
        atom_states = fg_out.enhanced_atom_states
        atom_fg_context = fg_out.atom_fg_context
        edge_initial = self.edge_input(f_bonds)
        edge_states = edge_initial
        edge_mask = torch.ones(edge_states.size(0), 1, dtype=edge_states.dtype, device=edge_states.device)
        if edge_mask.numel():
            edge_mask[0, 0] = 0.0

        enc_pair_rel = self.relation_encoder(sparse.pair_relation_features)
        enc_pair_states = self._init_pair_states(
            atom_states, atom_fg_context, enc_pair_rel, sparse.enc_carrier_pairs
        )
        depth = max(1, int(self.config.get("depth", 1)))
        for layer_idx in range(depth):
            nei_a_message = index_select_ND(edge_states, a2b)
            a_message = nei_a_message.sum(dim=1)
            rev_message = edge_states[b2revb]
            edge_message = a_message[b2a] - rev_message
            edge_states = self.edge_gru(edge_initial, edge_message) * edge_mask
            edge_states = self.edge_dropout(edge_states)

            incoming = index_select_ND(edge_states, a2b).sum(dim=1)
            h_hat = self.atom_update(torch.cat([fg_out.enhanced_atom_states, incoming], dim=1))
            if self.encoder_pair_layers:
                pair_layer = self.encoder_pair_layers[min(layer_idx, len(self.encoder_pair_layers) - 1)]
                enc_pair_states = pair_layer(
                    enc_pair_states,
                    enc_pair_rel,
                    sparse.enc_carrier_pairs,
                    sparse.enc_bridge_index,
                    sparse.enc_bridge_mask,
                )
            atom_states = self._apply_pair_feedback(
                h_hat, enc_pair_states, enc_pair_rel, sparse.enc_carrier_pairs
            )

        return atom_states, edge_states, enc_pair_states, enc_pair_rel, fg_out

    def _decoder_initial_states(
        self,
        atom_states: torch.Tensor,
        atom_fg_context: torch.Tensor,
        enc_pairs: torch.Tensor,
        enc_pair_states: torch.Tensor,
        dec_pairs: torch.Tensor,
        dec_pair_rel: torch.Tensor,
    ) -> torch.Tensor:
        enc_lookup = {tuple(pair): idx for idx, pair in enumerate(enc_pairs.tolist())}
        initialized = self._init_pair_states(atom_states, atom_fg_context, dec_pair_rel, dec_pairs)
        if dec_pairs.numel() == 0:
            return initialized
        rows = []
        for idx, pair in enumerate(dec_pairs.tolist()):
            enc_idx = enc_lookup.get(tuple(pair))
            if enc_idx is None:
                rows.append(initialized[idx])
            else:
                rows.append(self.reuse_pair(enc_pair_states[enc_idx]))
        return torch.stack(rows, dim=0)

    def _fg_pair_context(self, atom_fg_context: torch.Tensor, unordered_pairs: torch.Tensor) -> torch.Tensor:
        if unordered_pairs.numel() == 0:
            return torch.zeros((0, self.hidden_size * 2), dtype=atom_fg_context.dtype, device=atom_fg_context.device)
        left = atom_fg_context[unordered_pairs[:, 0]]
        right = atom_fg_context[unordered_pairs[:, 1]]
        return torch.cat([left + right, torch.abs(left - right)], dim=1)

    def _gold_bond_pairs(self, targets: list[ContextualEditTarget] | None) -> set[tuple[int, int]]:
        if not targets:
            return set()
        return {target.gold_bond_pair for target in targets if target.edit_type == "bond" and target.gold_bond_pair is not None}

    def _proposal_targets(
        self,
        proposal_pairs: torch.Tensor,
        targets: list[ContextualEditTarget] | None,
    ) -> torch.Tensor:
        labels = torch.zeros((proposal_pairs.size(0),), dtype=torch.float32, device=proposal_pairs.device)
        gold = self._gold_bond_pairs(targets)
        if not gold or proposal_pairs.numel() == 0:
            return labels
        lookup = {tuple(pair): idx for idx, pair in enumerate(proposal_pairs.tolist())}
        for pair in gold:
            idx = lookup.get(tuple(sorted(pair)))
            if idx is not None:
                labels[idx] = 1.0
        return labels

    def _select_topk_pairs(self, sparse, proposal_logits: torch.Tensor) -> list[set[tuple[int, int]]]:
        selected: list[set[tuple[int, int]]] = []
        pair_topk = int(self.config.get("pair_topk", 64))
        for pair_start, pair_count in sparse.proposal_pair_scope:
            pairs = sparse.proposal_universe_pairs[pair_start: pair_start + pair_count]
            logits = proposal_logits[pair_start: pair_start + pair_count]
            if pair_count == 0:
                selected.append(set())
                continue
            k = min(pair_topk, int(pair_count))
            top_indices = torch.topk(logits, k=k).indices.detach().cpu().tolist()
            selected.append({tuple(map(int, pairs[idx].detach().cpu().tolist())) for idx in top_indices})
        return selected

    def _build_dynamic_decoder(
        self,
        graph_batch: GraphBatch,
        proposal_topk_pairs: list[set[tuple[int, int]]],
        targets: list[ContextualEditTarget] | None,
    ) -> None:
        sparse = graph_batch.sparse_metadata
        assert sparse is not None
        if graph_batch.mols is None:
            raise ValueError(
                "Prepared data lacks molecule metadata needed for dynamic contextual_2fwl decoding. "
                "Re-run prepare_data.py with --model_variant contextual_2fwl."
            )
        decoder_items = []
        enc_score_rows = sparse.enc_score_pairs.detach().cpu().tolist()
        for mol_idx, (atom_start, atom_count) in enumerate(sparse.atom_scope):
            atom_end = atom_start + atom_count
            enc_score_pairs = {
                tuple(map(int, pair))
                for pair in enc_score_rows
                if atom_start <= int(pair[0]) < atom_end and atom_start <= int(pair[1]) < atom_end
            }
            mol_targets = [targets[mol_idx]] if targets and mol_idx < len(targets) else None
            decoder_items.append(
                build_decoder_pair_metadata(
                    graph_batch.mols[mol_idx],
                    (graph_batch.fg_metadata or [])[mol_idx],
                    enc_score_pairs=enc_score_pairs,
                    proposal_topk_pairs=proposal_topk_pairs[mol_idx],
                    gold_bond_pairs=self._gold_bond_pairs(mol_targets),
                    training=self.training and mol_targets is not None,
                    atom_offset=atom_start,
                    pair_near_radius=self.config.get("pair_near_radius", 2),
                    pair_bridge_radius=self.config.get("pair_bridge_radius", 2),
                    pair_max_score_pairs_dec=self.config.get("pair_max_score_pairs_dec", 1024),
                    pair_max_carrier_pairs_dec=self.config.get("pair_max_carrier_pairs_dec", 2048),
                    pair_max_bridges_dec=self.config.get("pair_max_bridges_dec", 8),
                )
            )
        merged = merge_decoder_pair_metadata(decoder_items)
        device = sparse.enc_carrier_pairs.device
        for key, value in merged.items():
            if key == "diagnostics":
                sparse.diagnostics.update(value)
            elif isinstance(value, torch.Tensor):
                setattr(sparse, key, value.to(device))
            else:
                setattr(sparse, key, value)

    def _score_actions(
        self,
        atom_states: torch.Tensor,
        atom_fg_context: torch.Tensor,
        dec_pair_states: torch.Tensor,
        dec_pair_rel: torch.Tensor,
        graph_batch: GraphBatch,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        sparse = graph_batch.sparse_metadata
        assert sparse is not None
        pair_lookup = {tuple(pair): idx for idx, pair in enumerate(sparse.dec_carrier_pairs_base.tolist())}
        action_lengths: list[int] = []
        edit_scores: list[torch.Tensor] = []
        graph_vecs: list[torch.Tensor] = []

        for mol_idx, (atom_start, atom_count) in enumerate(sparse.atom_scope):
            pair_start, pair_count = sparse.action_pair_scope[mol_idx]
            candidate_pairs = sparse.unordered_dec_candidate_pairs[pair_start: pair_start + pair_count]
            bond_logits = []
            for i_abs, j_abs in candidate_pairs.tolist():
                q_ij_idx = pair_lookup.get((int(i_abs), int(j_abs)))
                q_ji_idx = pair_lookup.get((int(j_abs), int(i_abs)))
                if q_ij_idx is None or q_ji_idx is None:
                    q_ij = torch.zeros(self.pair_hidden_size, dtype=atom_states.dtype, device=atom_states.device)
                    q_ji = q_ij
                    pair_rel = torch.zeros(self.pair_relation_size, dtype=atom_states.dtype, device=atom_states.device)
                else:
                    q_ij = dec_pair_states[q_ij_idx]
                    q_ji = dec_pair_states[q_ji_idx]
                    pair_rel = dec_pair_rel[q_ij_idx]
                q_sym = self.bond_rep(torch.cat([q_ij + q_ji, torch.abs(q_ij - q_ji), q_ij * q_ji], dim=0))
                fg_sum = atom_fg_context[int(i_abs)] + atom_fg_context[int(j_abs)]
                fg_abs = torch.abs(atom_fg_context[int(i_abs)] - atom_fg_context[int(j_abs)])
                bond_logits.append(self.bond_linear(torch.cat([q_sym, pair_rel, fg_sum, fg_abs], dim=0)))
            if bond_logits:
                bond_block = torch.stack(bond_logits, dim=0).flatten()
            else:
                bond_block = torch.zeros((0,), dtype=atom_states.dtype, device=atom_states.device)

            atom_logits = []
            for atom_abs in range(atom_start, atom_start + atom_count):
                diag_idx = pair_lookup.get((atom_abs, atom_abs))
                q_diag = (
                    dec_pair_states[diag_idx]
                    if diag_idx is not None
                    else torch.zeros(self.pair_hidden_size, dtype=atom_states.dtype, device=atom_states.device)
                )
                atom_logits.append(
                    self.atom_linear(torch.cat([atom_states[atom_abs], q_diag, atom_fg_context[atom_abs]], dim=0))
                )
            atom_block = torch.stack(atom_logits, dim=0).flatten()
            mol_atom_states = atom_states[atom_start: atom_start + atom_count]
            graph_vec = mol_atom_states.sum(dim=0)
            if pair_count:
                pair_values = []
                for i_abs, j_abs in candidate_pairs.tolist():
                    pair_idx = pair_lookup.get((int(i_abs), int(j_abs)))
                    if pair_idx is not None:
                        pair_values.append(dec_pair_states[pair_idx])
                pair_pool = torch.stack(pair_values, dim=0).sum(dim=0) if pair_values else torch.zeros(
                    self.pair_hidden_size, dtype=atom_states.dtype, device=atom_states.device
                )
            else:
                pair_pool = torch.zeros(self.pair_hidden_size, dtype=atom_states.dtype, device=atom_states.device)
            stop = self.graph_linear(torch.cat([graph_vec, pair_pool], dim=0)).flatten()
            score = torch.cat([bond_block, atom_block, stop], dim=0)
            edit_scores.append(score)
            graph_vecs.append(graph_vec)
            action_lengths.append(int(score.numel()))

        sparse.action_vector_lengths = action_lengths
        return edit_scores, torch.stack(graph_vecs, dim=0)

    def compute_edit_scores(self, graph_batch: GraphBatch, targets: list[ContextualEditTarget] | None = None):
        if graph_batch.fg_metadata is None or graph_batch.sparse_metadata is None:
            raise ValueError(
                "Prepared data lacks sparse pair metadata / gold action metadata. Re-run prepare_data.py "
                "with --model_variant contextual_2fwl."
            )
        sparse = graph_batch.sparse_metadata
        atom_states, _edge_states, enc_pair_states, _enc_pair_rel, fg_out = self._encode_contextual_graph(graph_batch)
        atom_fg_context = fg_out.atom_fg_context
        proposal_pair_rel = self.relation_encoder(sparse.proposal_pair_relation_features)
        proposal_fg_context = self._fg_pair_context(atom_fg_context, sparse.proposal_universe_pairs)
        proposal_logits = self.proposal_head(
            atom_states,
            sparse.proposal_universe_pairs,
            proposal_pair_rel,
            proposal_fg_context,
        )
        proposal_targets = self._proposal_targets(sparse.proposal_universe_pairs, targets)
        if proposal_logits.numel():
            self.last_proposal_loss = F.binary_cross_entropy_with_logits(proposal_logits, proposal_targets)
        else:
            self.last_proposal_loss = atom_states.sum() * 0.0
        proposal_topk_pairs = self._select_topk_pairs(sparse, proposal_logits)
        self._build_dynamic_decoder(graph_batch, proposal_topk_pairs, targets)

        dec_pair_rel = self.relation_encoder(sparse.dec_pair_relation_features)
        dec_pair_states = self._decoder_initial_states(
            atom_states,
            atom_fg_context,
            sparse.enc_carrier_pairs,
            enc_pair_states,
            sparse.dec_carrier_pairs_base,
            dec_pair_rel,
        )
        dec_pair_states = self.decoder(
            dec_pair_states,
            dec_pair_rel,
            sparse.dec_carrier_pairs_base,
            sparse.dec_bridge_index_base,
            sparse.dec_bridge_mask_base,
        )
        gold_count = len(self._gold_bond_pairs(targets))
        topk_hits = 0
        inference_hits = 0
        if targets:
            for mol_idx, target in enumerate(targets):
                if target.edit_type != "bond" or target.gold_bond_pair is None:
                    continue
                gold = tuple(sorted(target.gold_bond_pair))
                if gold in proposal_topk_pairs[mol_idx]:
                    topk_hits += 1
                atom_start, atom_count = sparse.atom_scope[mol_idx]
                atom_end = atom_start + atom_count
                inference_pairs = proposal_topk_pairs[mol_idx] | {
                    tuple(sorted((int(i), int(j))))
                    for i, j in sparse.enc_score_pairs.tolist()
                    if i != j and atom_start <= int(i) < atom_end and atom_start <= int(j) < atom_end
                }
                if gold in inference_pairs:
                    inference_hits += 1
        self.last_diagnostics = {
            **sparse.diagnostics,
            **fg_out.diagnostics,
            "proposal_logits": proposal_logits.detach(),
            "proposal_bce": float(self.last_proposal_loss.detach().cpu().item()),
            "Recall_bond(S_topK)": float(topk_hits / gold_count) if gold_count else 1.0,
            "Recall_bond(S_dec_test_score)": float(inference_hits / gold_count) if gold_count else 1.0,
            "Recall_atom": 1.0,
        }
        return self._score_actions(atom_states, atom_fg_context, dec_pair_states, dec_pair_rel, graph_batch)

    def map_gold_target_indices(
        self,
        targets: list[ContextualEditTarget],
        graph_batch: GraphBatch,
    ) -> torch.LongTensor:
        sparse = graph_batch.sparse_metadata
        assert sparse is not None
        indices: list[int] = []
        for mol_idx, target in enumerate(targets):
            pair_start, pair_count = sparse.action_pair_scope[mol_idx]
            atom_start, atom_count = sparse.atom_scope[mol_idx]
            vector_length = pair_count * self.bond_outdim + atom_count * self.atom_outdim + 1
            if target.stop or target.edit_type == "stop":
                indices.append(vector_length - 1)
            elif target.edit_type == "bond":
                pair_lookup = {
                    tuple(row): idx
                    for idx, row in enumerate(
                        sparse.unordered_dec_candidate_pairs[pair_start: pair_start + pair_count].tolist()
                    )
                }
                gold_pair = tuple(sorted(target.gold_bond_pair or ()))
                if gold_pair not in pair_lookup:
                    raise ValueError(
                        f"Gold bond pair {target.atom_maps} is absent from dynamic contextual_2fwl decoder candidates."
                    )
                indices.append(pair_lookup[gold_pair] * self.bond_outdim + int(target.bond_class_index or 0))
            elif target.edit_type == "atom":
                if target.gold_atom_index is None:
                    raise ValueError("Gold atom action is missing an absolute atom index.")
                local_atom = int(target.gold_atom_index) - atom_start
                if local_atom < 0 or local_atom >= atom_count:
                    raise ValueError(f"Gold atom {target.atom_maps} is outside the current molecule scope.")
                indices.append(pair_count * self.bond_outdim + local_atom * self.atom_outdim + int(target.atom_class_index or 0))
            else:
                raise ValueError(f"Unsupported contextual edit target type: {target.edit_type}")
        return torch.tensor(indices, dtype=torch.long, device=sparse.enc_carrier_pairs.device)

    def decode_action(self, mol, graph_batch: GraphBatch, edit_logits: torch.Tensor, idx: torch.Tensor | int):
        sparse = graph_batch.sparse_metadata
        assert sparse is not None
        idx_int = int(idx.item() if hasattr(idx, "item") else idx)
        pair_start, pair_count = sparse.action_pair_scope[0]
        atom_start, atom_count = sparse.atom_scope[0]
        bond_block = pair_count * self.bond_outdim
        if idx_int == int(edit_logits.numel()) - 1:
            return "Terminate", []
        if idx_int < bond_block:
            pair_pos = idx_int // self.bond_outdim
            edit_idx = idx_int % self.bond_outdim
            i_abs, j_abs = sparse.unordered_dec_candidate_pairs[pair_start + pair_pos].tolist()
            local_i = int(i_abs) - atom_start
            local_j = int(j_abs) - atom_start
            a1 = mol.GetAtomWithIdx(local_i).GetAtomMapNum() or (local_i + 1)
            a2 = mol.GetAtomWithIdx(local_j).GetAtomMapNum() or (local_j + 1)
            return self.bond_vocab.get_elem(edit_idx), sorted([a1, a2])
        atom_idx_flat = idx_int - bond_block
        local_atom = atom_idx_flat // self.atom_outdim
        edit_idx = atom_idx_flat % self.atom_outdim
        a1 = mol.GetAtomWithIdx(local_atom).GetAtomMapNum() or (local_atom + 1)
        return self.atom_vocab.get_elem(edit_idx), a1
