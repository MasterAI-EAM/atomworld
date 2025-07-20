import pytest
import numpy as np
from ase import Atoms
from AtomWorldBench.atom_world.motifs.site import SiteMotif
from AtomWorldBench.atom_world.motif_detectors.site import SiteDetector
from AtomWorldBench.tests.test_motif_detectors.common_detector_test import BaseMotifDetectorTest


class TestSiteDetector(BaseMotifDetectorTest):
    @pytest.fixture
    def detector(self):
        return SiteDetector(cutoff=3.5, symbols=["Li", "Cl"])

    def test_detect_around_frac_coords_len_error(self, simple_atoms, detector):
        frac_coords = np.array([[0.1, 0.1, 0.1], [2.1, 0.1, 0.1]])
        with pytest.raises(ValueError):
            detector.detect_around_frac_coords(simple_atoms, frac_coords)
        
    def test_detect_around_frac_coords_type_x_error(self, simple_atoms, detector):
        frac_coords = np.array([0.1, 0.1, 0.1])
        simple_atoms += Atoms("X", positions=[[0.1, 0.1, 0.1]], cell=simple_atoms.cell, pbc=simple_atoms.pbc)
        with pytest.raises(ValueError):
            detector.detect_around_frac_coords(simple_atoms, frac_coords)
    
    def test_detect_around_frac_coords_valid(self, simple_atoms, detector):
        frac_coords = np.array([0.1, 0.1, 0.1])
        detected_sites = detector.detect_around_frac_coords(simple_atoms, frac_coords)
        
        assert len(detected_sites) == 1
        assert all(isinstance(site, Atoms) for site in detected_sites)
        assert np.allclose(detected_sites[0].get_positions(), [[0, 0, 0]])
        # assert np.allclose(detected_sites[1].get_positions(), [[2, 0, 0]])

    def test_detect_all(self, simple_atoms, detector):
        all_motif = detector.detect_all(simple_atoms)

        assert all_motif[0].get_chemical_symbols()[0] == 'Li'
        assert all_motif[1].get_chemical_symbols()[0] == 'Na'
        assert all_motif[2].get_chemical_symbols()[0] == 'Cl'