import pytest
import numpy as np
from ase import Atoms
from AtomWorldBench.atom_world.motifs.site import SiteMotif

class TestSiteMotif:
    @pytest.fixture
    def single_atom_motif(self):
        atoms = Atoms('H', positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=False)
        return SiteMotif(atoms)


    def test_raises_on_many_atoms(self):
        atoms = Atoms('HO', positions=[[0, 0, 0], [1, 1, 1]], cell=[2, 2, 2], pbc=True)
        with pytest.raises(ValueError, match="SiteMotif can only be initialized with a single index."):
            return SiteMotif(atoms, indices=[3, 2])
