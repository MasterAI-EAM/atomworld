import pytest
import numpy as np
from ase import Atoms
from AtomWorldBench.atom_world.motifs.site import SiteMotif
from AtomWorldBench.atom_world.motif_detectors.site import SiteDetector
from AtomWorldBench.tests.test_motif_detectors.common_detector_test import BaseMotifDetectorTest


class TestSiteDetector(BaseMotifDetectorTest):
    @pytest.fixture
    def simple_atoms(self):
        atoms = Atoms(
            symbols=["Li", "Na", "Cl"],
            positions=[[0, 0, 0], [2, 0, 0], [15, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
            charges=[1, 3, -1]
        )
        return atoms

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

        assert np.allclose(all_motif[0].positions[0], [0.,0.,0.])
        assert np.allclose(all_motif[1].positions[0], [5,0,0])

        detector.wrap = False
        all_motif = detector.detect_all(simple_atoms)
        assert np.allclose(all_motif[1].positions[0], [15,0,0])

    def test_detect_all_no_symbol(self, simple_atoms):
        new_detector = SiteDetector(cutoff=3.5)
        all_motif = new_detector.detect_all(simple_atoms)
        assert len(all_motif) == 3

    def test_detect_one(self, simple_atoms, detector):
        one_motif = detector.detect_one(simple_atoms)
        assert list(one_motif.composition.keys())[0] in ["Li +", "Cl -"]

        detector.wrap = False
        one_motif = detector.detect_one(simple_atoms)
        assert list(one_motif.composition.keys())[0] in ["Li +", "Cl -"]