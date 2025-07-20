from abc import abstractmethod
import numpy as np
import pytest
from ase import Atoms


class BaseMotifDetectorTest:

    @pytest.fixture
    def simple_atoms(self):
        atoms = Atoms(
            symbols=["Li", "Na", "Cl"],
            positions=[[0, 0, 0], [2, 0, 0], [5, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
            charges=[1, 3, -1]
        )
        return atoms
    
    @pytest.fixture
    @abstractmethod
    def detector(self):
        pass
    
    @abstractmethod
    def test_detect_around_frac_coords(self, simple_atoms, detector):
        pass

    



