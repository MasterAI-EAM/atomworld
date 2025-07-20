from abc import abstractmethod
import numpy as np
import pytest
# from AtomWorldBench.atom_world.motifs.utils import get_species_string

class BaseMotifTests:

    @pytest.fixture
    @abstractmethod
    def motif(self):
        pass
    
    # @pytest.fixture
    # def expected_atoms(self):
    #     """Return the ASE Atoms object used to construct the motif."""
    #     raise NotImplementedError("Subclasses must provide `expected_atoms`")


    # implemented like this for clarity
    @pytest.fixture
    @abstractmethod
    def expected_frac_coords(self):
        pass
    
    @pytest.fixture
    @abstractmethod
    def expected_radius(self):
        pass
    
    @pytest.fixture
    @abstractmethod
    def expected_cart_coords(self):
        pass
    
    @pytest.fixture
    @abstractmethod
    def expected_cell_offsets(self):
        pass
    
    @pytest.fixture
    @abstractmethod
    def expected_centroid(self):
        """Return a dict {True: frac_centroid, False: cart_centroid}."""
        pass
    
    @pytest.fixture
    @abstractmethod
    def expected_strings(self):
        """
        Dict of strings: name, species_strings, etc.
        """
        pass
    
    @pytest.fixture
    @abstractmethod
    def expected_edge_lengths(self):
        pass

    @pytest.fixture
    @abstractmethod
    def expected_indices(self):
        pass
    
    def test_strings(self, motif, expected_strings):
        for attr, expected_value in expected_strings.items():
            actual_value = getattr(motif, attr)
            assert actual_value == expected_value, f"{attr} mismatch: expected {expected_value}, got {actual_value}"
    
    def test_radius(self, motif, expected_radius):
        assert motif.radius == pytest.approx(expected_radius, abs=1e-3)
        
    def test_edge_lengths(self, motif, expected_edge_lengths):
        assert motif.edge_lengths == expected_edge_lengths

    def test_cell_offsets(self, motif, expected_cell_offsets):
        assert np.allclose(motif.cell_offsets, expected_cell_offsets)

    def test_cart_coords(self, motif, expected_cart_coords):
        assert np.allclose(motif.cart_coords, expected_cart_coords)

    def test_frac_coords(self, motif, expected_frac_coords):
        assert np.allclose(motif.frac_coords, expected_frac_coords)
    
    def test_indices(self, motif, expected_indices):
        assert motif.indices == expected_indices

    @pytest.mark.parametrize("fractional", [False, True])
    def test_centroid(self, motif, expected_centroid, fractional):
        expected = expected_centroid[fractional]
        result = motif.get_centroid(fractional=fractional)
        assert np.allclose(result, expected)


# def test_get_species_string():
#     with pytest.raises(TypeError, match="Charge must be an integer or None."):
#         return get_species_string('Ar', '0.5')
    
#     assert get_species_string('A', -2) == 'A 2-'
