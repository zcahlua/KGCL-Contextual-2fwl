from kgcl_retro.chemistry.functional_groups import load_functional_group_resources  # Explanation: imports the packaged functional-group asset loader.


def test_functional_group_assets_load():  # Explanation: verifies packaged KG functional-group assets can be loaded.
    resources = load_functional_group_resources("KGembedding")  # Explanation: loads the default functional-group embedding set.

    assert len(resources.names) > 0  # Explanation: checks that at least one functional group was read.
    assert len(resources.smarts) == len(resources.names)  # Explanation: checks that SMARTS patterns align with names.
    assert "Phenyl" in resources.names  # Explanation: checks for a known functional group from the asset file.
    assert "Phenyl" in resources.embeddings  # Explanation: checks that the known functional group has an embedding.
