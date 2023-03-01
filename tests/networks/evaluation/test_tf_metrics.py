import tensorflow as tf
import numpy as np
from pytest import mark

from midap.networks.evaluation import tf_metrics


# Tests
#######

def test_toggle_metrics():
    """
    A very simple test to see if we can init the metric
    """

    toggle = tf_metrics.ToggleMetrics()
    toggle = tf_metrics.ToggleMetrics(toggle_metrics=["custom_metric"])


@mark.usefixtures("img_and_seg")
def test_roi_accuracy(img_and_seg):
    """
    Tests the ROI accuracy metric with the test image and segmentation
    :param img_and_seg: A fixture that creates a test image and a test segmentations
    """

    # unpack
    img, seg = img_and_seg

    # to tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    seg = tf.convert_to_tensor(seg, dtype=tf.float32)

    # init the metric
    roi_accuracy = tf_metrics.ROIAccuracy()

    # update state
    roi_accuracy.update_state(seg, img)

    # See fixture to see that accuracy is 1.0 on ROI
    assert np.isclose(roi_accuracy.result().numpy(), 1.0)

    # Another update should not change the accuracy
    roi_accuracy.update_state(seg, img)
    assert np.isclose(roi_accuracy.result().numpy(), 1.0)

    # reset the state -> nan accuracy
    roi_accuracy.reset_state()
    assert np.isnan(roi_accuracy.result().numpy())


@mark.usefixtures("img_and_seg")
def test_average_precision(img_and_seg):
    """
    Tests the AveragePrecision metric with the test image and segmentation
    :param img_and_seg: A fixture that creates a test image and a test segmentations
    """

    # unpack
    img, seg = img_and_seg

    # to tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    seg = tf.convert_to_tensor(seg, dtype=tf.float32)

    # init the metric
    average_precision = tf_metrics.AveragePrecision(threshold=0.99)

    # update state
    average_precision.update_state(seg, img)

    # update should not have worked be because metric is off
    assert np.isnan(average_precision.result().numpy())

    # init the metric
    average_precision = tf_metrics.AveragePrecision(threshold=0.99, on_start=True)

    # update state
    average_precision.update_state(seg, img)

    # now we should get something (see fixture for value)
    assert np.isclose(average_precision.result().numpy(), 399/400)

    # another update should increase the true positives to 2*399
    # update state
    average_precision.update_state(seg, img)

    # now we should get something (see fixture for value)
    assert np.isclose(average_precision.tp.numpy(), 2*399)

    # reset and we get nan again
    average_precision.reset_state()
    assert np.isnan(average_precision.result().numpy())
