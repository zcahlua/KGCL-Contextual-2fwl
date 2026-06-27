from kgcl_retro.config.schema import (
    CONTEXTUAL_2FWL_ALIASES,
    apply_model_variant_defaults,
    normalize_model_variant,
)
from kgcl_retro.cli.train import build_arg_parser, build_model_config


def test_contextual_variant_aliases_normalize():
    assert normalize_model_variant("kgcl") == "kgcl"
    for alias in CONTEXTUAL_2FWL_ALIASES:
        assert normalize_model_variant(alias) == "contextual_2fwl"


def test_contextual_variant_enables_required_flags():
    config = apply_model_variant_defaults({"model_variant": "contextual-fg-kgcl-2fwl"})

    assert config["model_variant"] == "contextual_2fwl"
    assert config["use_contextual_fg"] is True
    assert config["use_sparse_2fwl"] is True
    assert config["global_action_softmax"] is True


def test_baseline_variant_preserves_default_flags():
    config = apply_model_variant_defaults({"model_variant": "kgcl"})

    assert config["model_variant"] == "kgcl"
    assert config["use_contextual_fg"] is False
    assert config["use_sparse_2fwl"] is False


def test_train_parser_builds_contextual_config():
    parser = build_arg_parser()
    args = parser.parse_args(["--model_variant", "contextual_fg_2fwl"]).__dict__
    config = build_model_config(args)

    assert config["model_variant"] == "contextual_2fwl"
    assert config["fg_context_radius"] == 1
    assert config["pair_topk"] == 64
