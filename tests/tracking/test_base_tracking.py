import pytest
from midap.tracking.base_tracking import Tracking


def test_base_cutout():
    """
    Tests the SegmentationPredictor abstract base class
    """

    with pytest.raises(TypeError):
        _ = Tracking(
            imgs=None,
            segs=None,
            model_weights=None,
            input_size=None,
            target_size=None,
            connectivity=1,
        )
