import pytest
from unittest.mock import Mock
import numpy as np
from AtomWorldBench.atom_world.motif_description_styles.coord import CoordDescriptionStyle
from AtomWorldBench.atom_world.motifs.cluster import ClusterMotif
from AtomWorldBench.atom_world.motifs import motif_factory
from ase import Atoms


@pytest.fixture
def motif():
    atoms = Atoms(
        symbols=["Na", "Cl", "K"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        cell=np.eye(3) * 5.0,
        pbc=True,
        charges=[1, -1, 2]
    )
    return motif_factory("cluster", atoms, indices=np.array([0, 1, 8]))

@pytest.mark.parametrize("flavor, precision, center", [
    ("fractional", 4, False),
    ("fractional", 4, True),
    ("cartesian", 4, False),
    ("cartesian", 4, True)
])
def test_describe(flavor, precision, center, motif):
    style = CoordDescriptionStyle(flavor, precision, center)

    result = style.describe(motif)

    assert result.startswith(f"a triplet of atoms/species Na +, Cl -, K 2+ with {flavor} coordinates")




