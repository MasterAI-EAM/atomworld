import pytest
from ase import Atoms
from AtomWorldBench.atom_world.motifs.site import SiteMotif
from AtomWorldBench.atom_world.motifs import motif_factory

class TestSiteMotif:
    cls = SiteMotif
    
    @pytest.fixture
    def single_atom_motif(self):
        atoms = Atoms('H', positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=False)
        return motif_factory("site", atoms)


    def test_raises_on_many_atoms(self):
        atoms = Atoms('HO', positions=[[0, 0, 0], [1, 1, 1]], cell=[2, 2, 2], pbc=True)
        with pytest.raises(ValueError, match="SiteMotif must contain exactly one site, but got 2 sites."):
            return SiteMotif(atoms, indices=[3, 2])

    def test_get_name(self):
        atoms = Atoms('Pt', positions=[[10,0,0]],cell=[1,1,1], charges=[20])
        assert SiteMotif(atoms).name == 'one Pt 20+ ion'
        atoms = Atoms('Pt', positions=[[10,0,0]],cell=[1,1,1])
        assert SiteMotif(atoms).name == 'one Pt atom'