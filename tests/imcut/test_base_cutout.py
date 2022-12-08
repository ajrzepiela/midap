import pytest
from midap.imcut.base_cutout import CutoutImage

def test_base_cutout():
    """
    Tests the CutoutImage abstract base class
    """

    with pytest.raises(TypeError):
        _ = CutoutImage(paths=None)
