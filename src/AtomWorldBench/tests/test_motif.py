import numpy as np
import pytest
from ase import Atoms
from AtomWorldBench.atom_world.motifs.bond import BondMotif
from AtomWorldBench.atom_world.motifs.base import BaseMotif
from AtomWorldBench.atom_world.motifs.cluster import ClusterMotif


@pytest.fixture
def simple_atoms_pair():
    return Atoms(
        symbols=["Na", "Cl"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=np.eye(3) * 3.0,
        pbc=True,
        charges=[1.0, -1.0]
    )

@pytest.fixture
def simple_atoms_triplet():
    return Atoms(
        symbols=["Na", "Cl", "K"],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        cell=np.eye(3) * 5.0,
        pbc=True,
        charges=[1, -1, 2]
    )

@pytest.fixture
def bond_motif(simple_atoms_pair):
    return BondMotif.from_atoms(simple_atoms_pair)

@pytest.fixture
def cluster_motif(simple_atoms_triplet):
    return ClusterMotif.from_atoms(simple_atoms_triplet, indices=[0,1,2])

@pytest.fixture
def simple_atoms():
    # ASE atoms: H at (0,0,0), O at (0.5,0.5,0.5)
    return Atoms('HO', 
                 positions=[[0,0,0], [1,1,1]],
                 cell=np.eye(3)*2.0, pbc=True, 
                 charges=[0,0])

# bond motif
def test_bond_motif_base_properties(bond_motif):
    assert bond_motif.name == "a bond between Na + and Cl -"
    assert bond_motif.radius == pytest.approx(0.5, abs=1e-3)
    assert bond_motif.cart_coords.shape == (2, 3)
    assert bond_motif.species_strings == ["Na +", "Cl -"]


# cluster motif
def test_cluster_motif_base_properties(cluster_motif):
    assert cluster_motif.indices.tolist() == [0,1,2]
    assert cluster_motif.species_strings == ['Na +', 'Cl -', 'K 2+']
    assert cluster_motif.radius > 0
    assert cluster_motif.edge_lengths[(0,1)] > 0


def test_cluster_motif_coords(cluster_motif):
    assert np.allclose(cluster_motif.cart_coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(cluster_motif.frac_coords, [[0,0,0], [0.2,0.0,0.0],[0.0, 0.2, 0.0]])
    assert np.allclose(cluster_motif.cell_offsets, [[0,0,0],[0,0,0], [0,0,0]])
    assert isinstance(cluster_motif.get_centroid(), np.ndarray)
    assert cluster_motif.get_centroid().shape == (3,)
    # assert isinstance(motif.describe("index"), str)  # Not implemented yet, so ignore first

