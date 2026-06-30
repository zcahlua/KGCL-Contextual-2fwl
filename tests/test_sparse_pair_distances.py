import pytest

Chem = pytest.importorskip("rdkit.Chem")

from kgcl_retro.chemistry.contextual_fg import INF_DISTANCE
from kgcl_retro.chemistry.sparse_pair_builder import _distances


def _old_floyd_warshall(mol):
    n_atoms = mol.GetNumAtoms()
    distances = [[INF_DISTANCE] * n_atoms for _ in range(n_atoms)]
    for i in range(n_atoms):
        distances[i][i] = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        distances[i][j] = 1
        distances[j][i] = 1
    for k in range(n_atoms):
        for i in range(n_atoms):
            via_k = distances[i][k]
            if via_k == INF_DISTANCE:
                continue
            for j in range(n_atoms):
                candidate = via_k + distances[k][j]
                if candidate < distances[i][j]:
                    distances[i][j] = candidate
    return distances


def test_distances_chain():
    mol = Chem.MolFromSmiles("CCCC")

    distances = _distances(mol)

    assert distances[0][0] == 0
    assert distances[0][1] == 1
    assert distances[0][2] == 2
    assert distances[0][3] == 3


def test_distances_ring():
    mol = Chem.MolFromSmiles("C1CCCCC1")

    distances = _distances(mol)

    assert distances[0][1] == 1
    assert distances[0][5] == 1
    assert distances[0][3] == 3


def test_distances_branch():
    mol = Chem.MolFromSmiles("CC(C)O")

    distances = _distances(mol)

    assert distances[0][1] == 1
    assert distances[0][2] == 2
    assert distances[2][3] == 2


def test_distances_disconnected_molecule():
    mol = Chem.MolFromSmiles("CC.O")

    distances = _distances(mol)

    assert distances[0][1] == 1
    assert distances[0][2] == INF_DISTANCE
    assert distances[2][0] == INF_DISTANCE


def test_bfs_matches_old_floyd_warshall_on_tiny_mols():
    for smiles in ["C", "CC", "CCO", "C1CC1", "CC.O", "CC(C)O"]:
        mol = Chem.MolFromSmiles(smiles)

        assert _distances(mol) == _old_floyd_warshall(mol)
