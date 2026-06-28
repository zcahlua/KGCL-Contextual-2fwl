import pytest

pytest.importorskip("torch")

from kgcl_retro.cli.train import test as evaluate


class _AlwaysCorrectModel:
    def eval(self):
        return None

    def predict(self, _prod_smi, rxn_class=None):
        return ["edit"], [[1, 2]]


def test_validation_accuracy_denominator_counts_molecules_not_batches():
    valid_data = [
        (["A", "B"], [["edit"], ["edit"]], [[[1, 2]], [[1, 2]]], None),
        (["C"], [["edit"]], [[[1, 2]]], None),
    ]

    valid_acc, valid_first_step_acc = evaluate(_AlwaysCorrectModel(), valid_data)

    assert valid_acc == 1.0
    assert valid_first_step_acc == 1.0
