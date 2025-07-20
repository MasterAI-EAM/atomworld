import pytest
import numpy as np
from ase import Atoms
from AtomWorldBench.atom_world.motifs.bond import BondMotif
from AtomWorldBench.tests.test_motifs.common_test import BaseMotifTests 
from AtomWorldBench.atom_world.motifs import motif_factory

class TestBondMotif(BaseMotifTests):
    @pytest.fixture
    def single_atom_motif(self):
        atoms = Atoms('H', positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=False)
        return BondMotif(atoms)

    @pytest.fixture
    def motif(self):
        atoms = Atoms('HO', positions=[[0, 0, 0], [1, 1, 1]], cell=[2, 2, 2], pbc=True)
        return motif_factory("bond", atoms, indices=[0, 1])
    
    @pytest.fixture
    def expected_frac_coords(self):
        return np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    @pytest.fixture
    def expected_cart_coords(self):
        return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    @pytest.fixture
    def expected_radius(self):
        return np.sqrt(0.75)

    @pytest.fixture
    def expected_cell_offsets(self):
        return np.array([[0, 0, 0], [0, 0, 0]])

    @pytest.fixture
    def expected_centroid(self):
        return {
            True: np.array([0.25, 0.25, 0.25]),
            False: np.array([0.5, 0.5, 0.5])
        }

    @pytest.fixture
    def expected_edge_lengths(self):
        return {(0,1): np.sqrt(3)}
    
    @pytest.fixture
    def expected_indices(self):
        return [0, 1]

    @pytest.fixture
    def expected_strings(self):
        return {
            "name": "a bond between H and O",
            "species_strings": ["H", "O"]
        }
