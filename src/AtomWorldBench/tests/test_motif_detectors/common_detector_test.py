from abc import abstractmethod
import numpy as np
import pytest
from ase import Atoms


class BaseMotifDetectorTest:

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

    



