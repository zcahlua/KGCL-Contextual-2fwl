from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kgcl_retro.data.collate import GraphBatch
from kgcl_retro.models.contextual_fg import ContextualFGEncoder
from kgcl_retro.models.encoder import MPNEncoder
from kgcl_retro.models.sparse_pair import (
    CandidateProposalHead,
    PairRelationEncoder,
    Sparse2FWLDecoder,
    SparsePairLayer,
)


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

        self.encoder = MPNEncoder(
            atom_fdim=config["n_atom_feat"],
            bond_fdim=config["n_bond_feat"],
            hidden_size=self.hidden_size,
            depth=config["depth"],
            dropout=config["dropout_mpn"],
            atom_message=config["atom_message"],
        )
        self.fg_encoder = ContextualFGEncoder(
            atom_hidden_size=self.hidden_size,
            fg_hidden_size=config.get("fg_hidden_size", self.hidden_size),
            fg_layers=config.get("fg_layers", 2),
            fg_use_boundary_pool=config.get("fg_use_boundary_pool", True),
            fg_use_distance_bias=config.get("fg_use_distance_bias", True),
            fg_use_membership_bias=config.get("fg_use_membership_bias", True),
            kg_embedding_size=config.get("kg_embedding_size", 256),
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
        self.proposal_head = CandidateProposalHead(self.hidden_size)

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

    def compute_edit_scores(self, graph_batch: GraphBatch):
        if graph_batch.fg_metadata is None or graph_batch.sparse_metadata is None:
            raise ValueError(
                "Prepared data lacks sparse pair metadata. Re-run prepare_data.py "
                "with --model_variant contextual_2fwl."
            )
        base_tensors = graph_batch.base_tensors
        atom_scope, _bond_scope = graph_batch.scopes
        atom_states = self.encoder(base_tensors, mask=None)
        fg_out = self.fg_encoder(atom_states, graph_batch.fg_metadata, atom_scope)
        atom_states = fg_out.enhanced_atom_states
        atom_fg_context = fg_out.atom_fg_context
        sparse = graph_batch.sparse_metadata

        enc_pair_rel = self.relation_encoder(sparse.pair_relation_codes)
        enc_pair_states = self._init_pair_states(
            atom_states, atom_fg_context, enc_pair_rel, sparse.enc_carrier_pairs
        )
        for layer in self.encoder_pair_layers:
            enc_pair_states = layer(
                enc_pair_states,
                enc_pair_rel,
                sparse.enc_carrier_pairs,
                sparse.enc_bridge_index,
                sparse.enc_bridge_mask,
            )
        atom_states = self._apply_pair_feedback(
            atom_states, enc_pair_states, enc_pair_rel, sparse.enc_carrier_pairs
        )

        dec_pair_rel = self.relation_encoder(sparse.dec_pair_relation_codes)
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
        proposal_logits = self.proposal_head(atom_states, sparse.unordered_dec_candidate_pairs)
        self.last_diagnostics = {
            **sparse.diagnostics,
            **fg_out.diagnostics,
            "proposal_logits": proposal_logits.detach(),
        }
        return self._score_actions(atom_states, atom_fg_context, dec_pair_states, dec_pair_rel, graph_batch)

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
