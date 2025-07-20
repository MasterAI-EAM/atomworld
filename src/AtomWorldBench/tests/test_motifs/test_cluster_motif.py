import pytest
import numpy as np
from ase import Atoms
from AtomWorldBench.atom_world.motifs.cluster import ClusterMotif
from AtomWorldBench.tests.test_motifs.common_test import BaseMotifTests 
from AtomWorldBench.atom_world.motifs import motif_factory
from unittest.mock import MagicMock


class TestClusterMotif(BaseMotifTests):

    @pytest.fixture
    def single_atom_motif(self):
        atoms = Atoms('H', positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=False)
        return ClusterMotif.from_atoms(atoms)
    
    @pytest.fixture
    def motif(self):
        atoms = Atoms(
            symbols=["Na", "Cl", "K"],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            cell=np.eye(3) * 5.0,
            pbc=True,
            charges=[1, -1, 2]
        )
        return motif_factory("cluster", atoms, indices=[0, 1, 8])

    @pytest.fixture
    def expected_frac_coords(self):
        return np.array([[0.0, 0.0, 0.0], [1/5, 0.0, 0.0], [0.0,1/5,0.0]])

    @pytest.fixture
    def expected_cart_coords(self):
        return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    @pytest.fixture
    def expected_radius(self):
        return np.sqrt(5)/3

    @pytest.fixture
    def expected_cell_offsets(self):
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    @pytest.fixture
    def expected_centroid(self):
        return {
            True: np.array([1/15,1/15,0.0]),
            False: np.array([1/3, 1/3, 0.0])
        }

    @pytest.fixture
    def expected_edge_lengths(self):
        return {(0,1): 1.0, (0, 2):1.0, (1,2): np.sqrt(2)}

    @pytest.fixture
    def expected_indices(self):
        return [0, 1, 8]

    @pytest.fixture
    def expected_strings(self):
        return {
            "name": "a triplet of atoms/species Na +, Cl -, K 2+",
            "species_strings": ["Na +", "Cl -", "K 2+"]
        }
    
    @pytest.mark.parametrize("length,species,expected_prefix", [
        (2, ['H', 'O'], "a pair of atoms/species H, O"),
        (3, ['C', 'Bar', 'O'], "a triplet of atoms/species C, Bar, O"),
        (4, ['Na +', 'Cl', 'Na', 'Cl'], "a quadruplet of atoms/species Na +, Cl, Na, Cl"),
        (5, ['H']*5, "a quintuplet of atoms/species H, H, H, H, H"),
        (6, ['C']*6, "a sextuplet of atoms/species C, C, C, C, C, C"),
        (7, ['X']*7, "a 7-sites cluster of atoms/species X, X, X, X, X, X, X"),
    ])
    def test_cluster_get_default_name(self, length, species, expected_prefix):
        mock_motif = MagicMock()
        mock_motif.__len__.return_value = length
        mock_motif.species_strings = species
        result = ClusterMotif._get_default_name(mock_motif)
        assert result == expected_prefix

    def test_radius_single_atom(self, single_atom_motif):
        assert single_atom_motif.radius == 0.0

    def test_non_indices(self, single_atom_motif):
        assert single_atom_motif.indices == None

    def test_get_atoms(self, single_atom_motif):
        assert single_atom_motif.get_atoms() == Atoms('H', positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=False)


    def test_extend_with_valid_indices(self):
        atoms1 = Atoms('H', positions=[[0, 0, 0]])
        atoms2 = Atoms('O', positions=[[1, 1, 1]])
        motif1 = ClusterMotif.from_atoms(atoms1, indices=[0])
        motif2 = ClusterMotif.from_atoms(atoms2, indices=[1])

        motif1.name = "H" 
        motif1.extend(motif2)

        assert motif1._get_default_name() == "a pair of atoms/species H, O"
        assert len(motif1) == 2

    def test_extend_raises_on_inconsistent_indices(self):
        atoms1 = Atoms('H', positions=[[0, 0, 0]])
        atoms2 = Atoms('O', positions=[[1, 1, 1]])
        motif1 = ClusterMotif.from_atoms(atoms1, indices=[0])
        motif2 = ClusterMotif.from_atoms(atoms2)

        with pytest.raises(ValueError, match="Both motifs must have indices"):
            motif1.extend(motif2)

    def test_get_item(self, motif):
        atoms = Atoms(
            symbols=["Na"],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.eye(3) * 5.0,
            pbc=True,
            charges=[1]
        )
        assert motif[0] == ClusterMotif.from_atoms(atoms, indices=[0])

    def test_multiply_motif(self, motif):
        ref_atoms = Atoms(
            symbols=["Na", "Cl", "K"],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            cell=np.eye(3) * 5.0,
            pbc=True,
            charges=[1, -1, 2]
        )
        ref_motif = ClusterMotif(ref_atoms*2)
        assert motif * 2 == ref_motif

    def test_describe_coord_style(self, motif):
        result = motif.describe(style="coord", precision=3)
        assert isinstance(result, str)
        assert result == "a triplet of atoms/species Na +, Cl -, K 2+ with fractional coordinates ((0.000, 0.000, 0.000), (0.200, 0.000, 0.000), (0.000, 0.200, 0.000))"

    def test_describe_index_style(self, motif):
        result = motif.describe(style="index", precision=3)
        assert isinstance(result, str)
        assert result == "a triplet of atoms/species Na +, Cl -, K 2+ with site indices: 0, 1, 8 in the central reference cell."

    def test_describe_invalid_style(self, motif):
        with pytest.raises(ValueError, match="Description style 'wtf' is not allowed"):
            motif.describe(style="wtf")
