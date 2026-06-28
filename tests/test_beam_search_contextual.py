import pytest

pytest.importorskip("torch")
pytest.importorskip("rdkit.Chem")

from kgcl_retro.models import beam_search as beam_search_module
from kgcl_retro.models.beam_search import BeamSearch


def test_contextual_batch_graph_passes_pair_topk(monkeypatch):
    captured = {}

    def fake_get_batch_graphs(graphs, **kwargs):
        captured.update(kwargs)
        return "graph-batch"

    monkeypatch.setattr(beam_search_module, "get_batch_graphs", fake_get_batch_graphs)

    model = type(
        "Model",
        (),
        {
            "model_variant": "contextual_2fwl",
            "config": {
                "fg_context_radius": 1,
                "pair_near_radius": 2,
                "pair_bridge_radius": 2,
                "pair_max_score_pairs_enc": 512,
                "pair_max_score_pairs_dec": 1024,
                "pair_max_carrier_pairs_enc": 1024,
                "pair_max_carrier_pairs_dec": 2048,
                "pair_max_bridges_enc": 8,
                "pair_max_bridges_dec": 8,
                "pair_topk": 7,
            },
        },
    )()
    search = BeamSearch(model, step_beam_size=1, beam_size=1, use_rxn_class=False)

    assert search._batch_graph(object()) == "graph-batch"
    assert captured["pair_topk"] == 7
