import numpy as np
from pytest import mark

from midap.networks.evaluation import evaluation


# Tests
#######


@mark.usefixtures("img_and_seg")
def test_evaluate_accuracy(img_and_seg):
    """
    Tests the evaluate_accuracy method
    :param img_and_seg: A fixture that creates a test image and a test segmentations
    """

    # unpack
    img, seg = img_and_seg

    # eval
    accuracy = evaluation.evaluate_accuracy(img, seg, roi_only=False)

    assert accuracy.size == 1
    # we should have a 25 pixel missmatch (see fixture)
    test_acc = (img.size - 25) / img.size
    assert np.allclose(accuracy, test_acc)

    accuracy = evaluation.evaluate_accuracy(img, seg, roi_only=True)

    assert accuracy.size == 1
    # here everything is golden (see fixture)
    assert np.isclose(accuracy, 1.0)


@mark.usefixtures("img_and_seg")
def test_evaluate_bayes_stats(img_and_seg):
    """
    Tests the evaluate_bayes_stats method on a test image
    :param img_and_seg: A fixture that creates a test image and a test segmentations
    """

    # unpack
    img, seg = img_and_seg

    # eval
    tpr, fpr, tnr, fnr = evaluation.evaluate_bayes_stats(img, seg)

    # checks
    assert tpr.size == 1
    assert fpr.size == 1
    assert tnr.size == 1
    assert fnr.size == 1
    assert np.allclose(tpr, 1.0)
    # we should have a 25 pixel missmatch (see fixture)
    test_tnr = (img.size - 399 * 25 - 25) / (img.size - 399 * 25)
    assert np.allclose(tnr, test_tnr)
    assert np.allclose(fpr, 1.0 - test_tnr)
    assert np.allclose(fnr, 1.0 - tpr)
