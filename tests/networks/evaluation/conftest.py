import pytest
import numpy as np

@pytest.fixture()
def img_and_seg():
    """
    Creates a test image and test mask that almost fit perfectly
    :return: The test image and the test mask
    """

    # fake data
    test_img = np.ones((5, 5, 1))
    test_img = np.pad(test_img, [(5, 5), (5, 5), (0, 0)], mode='constant', constant_values=0.0)
    test_img = np.concatenate([test_img for i in range(20)], axis=1)
    test_img = np.concatenate([test_img for i in range(20)], axis=0)

    # the test mask is missing one cell
    test_mask = test_img.copy()
    test_mask[:10, :10, 0] = 0

    return test_img, test_mask