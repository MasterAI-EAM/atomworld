import pytest
import numpy as np
from ase import Atoms
from AtomWorldBench.atom_world.motif_detectors.bond import BondDetector
from AtomWorldBench.tests.test_motif_detectors.common_detector_test import BaseMotifDetectorTest
from collections import Counter

class TestBondDetector(BaseMotifDetectorTest):
    @pytest.fixture
    def simple_atoms(self):
        atoms = Atoms(
            symbols=["Li", "Na", "Li", "Cl"],
            positions=[[0, 0, 0], [2, 0, 0], [3, 0, 0], [1, 3, 0]],
            cell=[10, 10, 10],
            pbc=True
        )
        return atoms

    @pytest.fixture
    def detector(self):
        return BondDetector(cutoff=3.5, symbols=["Li", "Na"])
    
    def test_detect_one(self, simple_atoms, detector):
        one_motif = detector.detect_one(simple_atoms)
        acceptables = [Counter(['Na', 'Li']), Counter(['Li', 'Li'])]
        assert Counter(list(one_motif.symbols)) in acceptables

    @pytest.mark.parametrize("frac_coords,expected_len", [
        ([-0.2, -0.4, -0.2], 0), # very far that no atom inside the cutoff
        ([-0.2,0.0,0.0], 0),  # 1 atom inside
        ([0.1, 0.3, 0],  1), # 2
        ([0.1, 0, 0], 2), # 3
    ])
    def test_detect_around_frac_coords(self, simple_atoms, detector, frac_coords, expected_len):
        motifs = detector.detect_around_frac_coords(simple_atoms, frac_coords)

        assert len(motifs) == expected_len