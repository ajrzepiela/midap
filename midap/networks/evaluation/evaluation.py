import numpy as np

def evaluate_accuracy(prediction: np.ndarray, label: np.ndarray, roi_only=False):
    """
    Evaluates the accuracy of the prediction
    :param prediction: The prediction of the network, either a 3 dimensional array with WHC or a 4 dimensional array
                       with BWHC (batch, width, height, channel). Can be binary or probablility.
    :param label: The corresponding labels as binary array
    :param roi_only: If set to true, only the regions of interests are considered to evaluate the accuracy, i.e.
                     where the label is 1. If there is no label in the image the accuracy is 1 by default
    :return: An array containing the accuracy for each element in the batch (single entry for 3 dim inputs)
    """

    # check dimensions
    if prediction.ndim != 3 and prediction.ndim != 4:
        raise ValueError("prediction and segmentation should have shape BWHC or WHC!")
    if prediction.ndim == 3:
        prediction = prediction[None,...]
        label = label[None,...]

    # calculate the accuracy
    accuracy = np.zeros(len(prediction))
    for i, (p, l) in enumerate(zip(prediction, label)):
        # get the mask
        if roi_only:
            mask = l.astype(bool)
        else:
            mask = Ellipsis

        # filter
        p_masked = p[mask]
        if p_masked.size > 0:
            accuracy[i] = np.sum((p_masked > 0.5) == l[mask].astype(bool)) / p[mask].size
        else:
            # set the accuracy to 1 since there is no roi
            accuracy[i] = 1.0

    return accuracy


def evaluate_bayes_stats(prediction: np.ndarray, label: np.ndarray):
    """
    Evaluates the true positive rate, false positive rate, true negative rate and false negative rate
    :param prediction: The prediction of the network, either a 3 dimensional array with WHC or a 4 dimensional array
                       with BWHC (batch, width, height, channel). Can be binary or probablility.
    :param label: The corresponding labels as binary array
    :return: Arrays containing the positive rate, false positive rate, true negative rate and false negative rate
             for each element in the batch (single entry for 3 dim inputs)
    """

    # check dimensions
    if prediction.ndim != 3 and prediction.ndim != 4:
        raise ValueError("prediction and segmentation should have shape BWHC or WHC!")
    if prediction.ndim == 3:
        prediction = prediction[None, ...]
        label = label[None, ...]

    # calculate the accuracy
    tp = np.zeros(len(prediction))
    fp = np.zeros(len(prediction))
    tn = np.zeros(len(prediction))
    fn = np.zeros(len(prediction))
    for i, (p, l) in enumerate(zip(prediction, label)):
        # mask for positive and negative events
        positive_mask = l.astype(bool)
        negative_mask = ~positive_mask

        # calculate everything
        p_bool = p > 0.5
        tp[i] = np.sum(p_bool[positive_mask])
        fp[i] = np.sum(p_bool[negative_mask])
        tn[i] = np.sum(~p_bool[negative_mask])
        fn[i] = np.sum(~p_bool[positive_mask])


    # convert to rate (some can be undefined)
    tpr = np.where(tp + fn > 0, tp/(tp + fn), 1)
    tnr = np.where(tn + fp > 0, tn/(tn + fp), 1)
    return tpr, 1 - tnr, tnr, 1 - tpr
