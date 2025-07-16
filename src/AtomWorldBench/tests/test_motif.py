import numpy as np
import pytest
from ase import Atoms
from AtomWorldBench.atom_world.motifs.bond import BondMotif
from AtomWorldBench.atom_world.motifs.base import BaseMotif
from AtomWorldBench.atom_world.motifs.cluster import ClusterMotif

@pytest.fixture
def simple_atoms():
    return Atoms(
        symbols=["X"],
        positions=[[1.0, 0.0, 0.0]],
        cell=np.eye(3) * 2.0,
        pbc=True,
        charges=[0]
    )

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
    assert cluster_motif._get_default_name() == 'a triplet of atoms/species Na +, Cl -, K 2+'


def test_cluster_motif_coords(cluster_motif):
    assert np.allclose(cluster_motif.cart_coords, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(cluster_motif.frac_coords, [[0,0,0], [0.2,0.0,0.0],[0.0, 0.2, 0.0]])
    assert np.allclose(cluster_motif.cell_offsets, [[0,0,0],[0,0,0], [0,0,0]])
    assert isinstance(cluster_motif.get_centroid(), np.ndarray)
    assert cluster_motif.get_centroid().shape == (3,)
    assert np.allclose(cluster_motif.get_centroid(True), np.array([0.0666666,0.0666666,0.0]))
    # assert isinstance(motif.describe("index"), str)  # Not implemented yet, so ignore first

def test_motif_equality(cluster_motif):
    other_motif = ClusterMotif.from_atoms(cluster_motif.get_atoms(), indices=[0,1,2])
    assert cluster_motif == other_motif


# some other properties
def test_single_atom_case(simple_atoms):
    motif = ClusterMotif.from_atoms(simple_atoms)
    assert motif.radius == pytest.approx(0.0, abs=1e-6)
    assert motif.indices == None

def test_cannot_instantiate_base_motif(simple_atoms):
    with pytest.raises(TypeError):
        BaseMotif.from_atoms(simple_atoms)


def test_describe_valid_and_invalid(bond_motif):
    desc = bond_motif.describe("index")
    assert isinstance(desc, str)

    with pytest.raises(ValueError) as excinfo:
        bond_motif.describe("not_a_style")
    assert "Description style" in str(excinfo.value)
