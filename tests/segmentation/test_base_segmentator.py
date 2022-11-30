import pytest
from midap.segmentation.base_segmentator import SegmentationPredictor

def test_base_cutout():
    """
    Tests the SegmentationPredictor abstract base class
    """

    with pytest.raises(TypeError):
        _ = SegmentationPredictor(path_model_weights=None, postprocessing=None)