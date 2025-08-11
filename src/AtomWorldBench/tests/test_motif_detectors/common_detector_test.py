from abc import abstractmethod
import numpy as np
import pytest
from ase import Atoms
from AtomWorldBench.atom_world.motif_detectors import detector_factory
from AtomWorldBench.atom_world.motif_detectors.cluster import ClusterDetector
from AtomWorldBench.atom_world.motif_detectors.site import SiteDetector


class BaseMotifDetectorTest:
    name = None

    @pytest.fixture
    @abstractmethod
    def simple_atoms(self):
        pass
    
    @pytest.fixture
    @abstractmethod
    def detector(self):
        pass
    
    @abstractmethod
    def test_detect_around_frac_coords(self, simple_atoms, detector):
        pass

    @abstractmethod
    def test_detect_one(self, simple_atoms, detector):
        pass

    

@pytest.mark.parametrize("detector_name, cls, kwargs", [
    ("site", SiteDetector, {"cutoff": 3.0}),
    ("bond", ClusterDetector, {"cutoff": 3.0}),
    ("cluster", ClusterDetector, {"cutoff": 3.0, "max_cluster_size": 3, "max_cluster_radius": 3}),
])
def test_detector_factory_creates_correct_class(detector_name, cls, kwargs):
    detector = detector_factory(detector_name, **kwargs)
    assert isinstance(detector, cls)

