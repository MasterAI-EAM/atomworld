import numpy as np
import pytest
from AtomWorldBench.atom_world.motifs.bond import BondMotif

def test_bond_motif_basic_properties():
    positions = [[0, 0, 0], [1, 0, 0]]
    symbols = ['Na', 'Cl']
    charges= np.int_([1, -1])


    motif = BondMotif(symbols=symbols, positions=positions, charges=None)

    # 检查基本属性
    assert motif.name == "a bond between Na+ and Cl-"
    assert motif.radius == pytest.approx(0.5, abs=1e-3)
    assert motif.cart_coords.shape == (2, 3)
    assert motif.species_strings == ["Na+", "Cl-"]
