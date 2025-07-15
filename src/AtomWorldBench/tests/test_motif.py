import numpy as np
import pytest
from ase import Atoms
from AtomWorldBench.atom_world.motifs.bond import BondMotif
from AtomWorldBench.atom_world.motifs.base import BaseMotif
from AtomWorldBench.atom_world.motifs.cluster import ClusterMotif

@pytest.fixture
def simple_atoms():
    # ASE atoms: H at (0,0,0), O at (0.5,0.5,0.5)
    return Atoms('HO', 
                 positions=[[0,0,0], [1,1,1]],
                 cell=np.eye(3)*2.0, pbc=True, 
                 charges=[0,0])

def test_bond_motif_properties():
    positions = [[0, 0, 0], [1, 0, 0]]
    cell = np.eye(3) * 2.0
    symbols = ['Na', 'Cl']
    charges= [1, -1]

    motif = BondMotif(symbols=symbols, positions=positions, charges=charges, cell=cell)

    assert motif.name == "a bond between Na + and Cl -"
    assert motif.radius == pytest.approx(0.5, abs=1e-3)
    assert motif.cart_coords.shape == (2, 3)
    assert motif.species_strings == ["Na +", "Cl -"]


def test_cluster_motif_properties(simple_atoms):
    motif = ClusterMotif.from_atoms(simple_atoms, indices=[0,1])
    assert isinstance(motif, BaseMotif)
    assert motif.indices.tolist() == [0,1]
    assert motif.species_strings == ['H', 'O']
    assert np.allclose(motif.cart_coords, [[0,0,0], [1,1,1]])
    assert np.allclose(motif.frac_coords, [[0,0,0], [0.5,0.5,0.5]])
    assert np.allclose(motif.cell_offsets, [[0,0,0],[0,0,0]])
    assert motif.radius > 0
    assert motif.edge_lengths[(0,1)] > 0
    assert isinstance(motif.get_centroid(), np.ndarray)
    assert motif.get_centroid().shape == (3,)
    # assert isinstance(motif.describe("index"), str)