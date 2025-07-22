import pytest
import numpy as np
from ase import Atoms
from AtomWorldBench.utils.coord_utils import check_integer_translation



def test_check_integer_translation():
    atoms1 = Atoms("HO", positions=[[0,0,0], [1,1,1]], cell=[2,2,2])
    atoms2 = Atoms("HO", positions=[[0,0,0], [1,1,1]], cell=[2,2,2])
    atoms2.translate([2,4,8])

    frac1 = atoms1.get_scaled_positions(wrap=False)
    frac2 = atoms2.get_scaled_positions(wrap=False)

    results = check_integer_translation(frac1=frac1, frac2=frac2)
    assert results is not None
    assert np.array_equal(results[0], [0,1])
    assert np.array_equal(results[1], [0,1])
    assert np.array_equal(results[2], [1,2,4])

    results = check_integer_translation(frac1=np.array([[1,1,1]]), frac2=np.array([[1,1,1],[2,2,2]]))
    assert results is None

