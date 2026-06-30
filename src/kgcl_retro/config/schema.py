from __future__ import annotations

import argparse
import os
from typing import Any


CONTEXTUAL_2FWL_ALIASES = {
    "contextual_2fwl",
    "contextual_fg_2fwl",
    "contextual-fg-kgcl-2fwl",
}

CONTEXTUAL_CONFIG_DEFAULTS: dict[str, Any] = {
    "model_variant": "kgcl",
    "use_contextual_fg": False,
    "use_sparse_2fwl": False,
    "fg_context_radius": 1,
    "fg_hidden_size": 256,
    "kg_embedding_size": 256,
    "fg_layers": 2,
    "fg_pool": "sum",
    "fg_use_boundary_pool": True,
    "fg_use_distance_bias": True,
    "fg_use_membership_bias": True,
    "fg_null_token": True,
    "fg_max_instances": None,
    "pair_hidden_size": 256,
    "pair_relation_size": 128,
    "pair_enc_layers": 1,
    "pair_dec_layers": 1,
    "pair_near_radius": 2,
    "pair_bridge_radius": 2,
    "pair_topk": 64,
    "pair_max_score_pairs_enc": 512,
    "pair_max_score_pairs_dec": 1024,
    "pair_max_carrier_pairs_enc": 1024,
    "pair_max_carrier_pairs_dec": 2048,
    "pair_max_bridges_enc": 8,
    "pair_max_bridges_dec": 8,
    "pair_use_proposal": True,
    "pair_proposal_loss_weight": 0.1,
    "pair_diagnostics": True,
    "global_action_softmax": True,
    "contrastive_loss_weight": None,
}


def normalize_model_variant(model_variant: str | None) -> str:
    variant = (model_variant or "kgcl").strip()
    if variant == "kgcl":
        return "kgcl"
    if variant in CONTEXTUAL_2FWL_ALIASES:
        return "contextual_2fwl"
    known = ["kgcl"] + sorted(CONTEXTUAL_2FWL_ALIASES)
    raise ValueError(
        f"Unsupported model_variant '{model_variant}'. Expected one of: {', '.join(known)}"
    )


def _parse_optional_int(value: Any) -> int | None:
    if value in (None, "", "none", "None", "null", "NULL"):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, "", "none", "None", "null", "NULL"):
        return None
    return float(value)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_env_value(reference: Any, raw_value: str) -> Any:
    if isinstance(reference, bool):
        return _parse_bool(raw_value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(raw_value)
    if isinstance(reference, float):
        return float(raw_value)
    if reference is None:
        if raw_value.strip().lower() in {"", "none", "null"}:
            return None
        try:
            return int(raw_value)
        except ValueError:
            try:
                return float(raw_value)
            except ValueError:
                return raw_value
    return raw_value


def apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config)
    for key, reference in CONTEXTUAL_CONFIG_DEFAULTS.items():
        env_key = f"KGCL_{key.upper()}"
        if env_key in os.environ:
            merged[key] = _coerce_env_value(reference, os.environ[env_key])
    return apply_model_variant_defaults(merged)


def apply_model_variant_defaults(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(CONTEXTUAL_CONFIG_DEFAULTS)
    merged.update(config)
    merged["model_variant"] = normalize_model_variant(merged.get("model_variant"))

    if merged["model_variant"] == "contextual_2fwl":
        merged["use_contextual_fg"] = True
        merged["use_sparse_2fwl"] = True
        merged["global_action_softmax"] = True
    return merged


def add_contextual_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model_variant",
        type=str,
        default="kgcl",
        help=(
            "Model variant to run. Default 'kgcl' preserves original KGCL; "
            "'contextual_2fwl' enables sparse Contextual-FG-KGCL-2FWL."
        ),
    )
    parser.add_argument("--use_contextual_fg", action="store_true", default=False)
    parser.add_argument("--use_sparse_2fwl", action="store_true", default=False)
    parser.add_argument("--fg_context_radius", type=int, default=1)
    parser.add_argument("--fg_hidden_size", type=int, default=256)
    parser.add_argument("--kg_embedding_size", type=int, default=256)
    parser.add_argument("--fg_layers", type=int, default=2)
    parser.add_argument("--fg_pool", type=str, default="sum")
    parser.add_argument("--fg_use_boundary_pool", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fg_use_distance_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fg_use_membership_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fg_null_token", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fg_max_instances", type=_parse_optional_int, default=None)
    parser.add_argument("--pair_hidden_size", type=int, default=256)
    parser.add_argument("--pair_relation_size", type=int, default=128)
    parser.add_argument("--pair_enc_layers", type=int, default=1)
    parser.add_argument("--pair_dec_layers", type=int, default=1)
    parser.add_argument("--pair_near_radius", type=int, default=2)
    parser.add_argument("--pair_bridge_radius", type=int, default=2)
    parser.add_argument("--pair_topk", type=int, default=64)
    parser.add_argument("--pair_max_score_pairs_enc", type=int, default=512)
    parser.add_argument("--pair_max_score_pairs_dec", type=int, default=1024)
    parser.add_argument("--pair_max_carrier_pairs_enc", type=int, default=1024)
    parser.add_argument("--pair_max_carrier_pairs_dec", type=int, default=2048)
    parser.add_argument("--pair_max_bridges_enc", type=int, default=8)
    parser.add_argument("--pair_max_bridges_dec", type=int, default=8)
    parser.add_argument("--pair_use_proposal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pair_proposal_loss_weight", type=float, default=0.1)
    parser.add_argument("--pair_diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--global_action_softmax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contrastive_loss_weight", type=_parse_optional_float, default=None)
    return parser
