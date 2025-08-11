import pytest
from unittest.mock import Mock
import numpy as np
from AtomWorldBench.atom_world.motif_description_styles.index import IndexDescriptionStyle

@pytest.fixture
def style():
    return IndexDescriptionStyle()

def test_describe_none_indices(style):
    motif = Mock()
    motif.indices = None
    motif.cell_offsets = np.array([[0, 0, 0]])
    motif.name = "TestMotif"
    with pytest.raises(ValueError, match="Indices are not set"):
        style.describe(motif)

def test_describe_central_cell(style):
    motif = Mock()
    motif.indices = [1, 2, 3]
    motif.cell_offsets = np.zeros((3, 3), dtype=int)
    motif.name = "TestMotif"

    result = style.describe(motif)
    assert "TestMotif with site indices: 1, 2, 3 in the central reference cell." == result

def test_describe_with_offsets(style):
    motif = Mock()
    motif.indices = [0]
    motif.cell_offsets = np.array([[1, 0, -1.8]])
    motif.name = "MyMotif"

    result = style.describe(motif)
    assert result.startswith("MyMotif with site indices: 0 and cell offsets:")
    assert "(1, 0, -1)" in result