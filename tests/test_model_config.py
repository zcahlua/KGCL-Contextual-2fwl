from kgcl_retro.models import BeamSearch, KGCL  # Explanation: imports packaged model exports for smoke testing.


def test_model_classes_import():  # Explanation: verifies the model package exposes the main model and decoder.
    assert KGCL.__name__ == "KGCL"  # Explanation: checks the main neural model export.
    assert BeamSearch.__name__ == "BeamSearch"  # Explanation: checks the beam-search decoder export.
