import pytest
import numpy as np
from ase import Atoms
from AtomWorldBench.atom_world.motif_detectors.cluster import ClusterDetector
from AtomWorldBench.tests.test_motif_detectors.common_detector_test import BaseMotifDetectorTest


class TestClusterDetector(BaseMotifDetectorTest):
    @pytest.fixture
    def simple_atoms(self):
        atoms = Atoms(
            symbols=["Li", "Na", "Li", "Na", "Cl"],
            positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0,0,1], [1, 3, 0]],
            cell=[10, 10, 10],
            pbc=True
        )
        return atoms

    @pytest.fixture
    def detector(self):
        return ClusterDetector(cutoff=3.0, max_cluster_size=3, max_cluster_radius=3, symbols=["Li", "Na"])
    

    def test_detect_all(self, simple_atoms, detector):
        all_motifs = detector.detect_all(simple_atoms)
        assert len(all_motifs) == 10

    @pytest.mark.parametrize("frac_coords,expected_len", [
        ([-0.2, -0.4, -0.2], 0), # very far that no atom inside the cutoff
        ([-0.25,0.0,0.0], 4),  # 1 atom inside
        ([0.2, 0.25, 0.1],  1), # 2
        ([0.1, 0, 0], 10), # 3
    ])
    def test_detect_around_frac_coords(self, simple_atoms, detector, frac_coords, expected_len):
        detector.must_include_center = False
        all_motifs = detector.detect_around_frac_coords(simple_atoms, frac_coords)
        print(len(all_motifs))
        assert len(all_motifs) == expected_len
    
