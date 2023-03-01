from typing import List, Optional

import tensorflow as tf
from . import metrics
from skimage.measure import label


class ToggleMetrics(tf.keras.callbacks.Callback):
    """
    This callback makes it possible to evaluate some metrics only for validation sets during the training
    On test begin (i.e. when evaluate() is called or validation data is run during fit()) toggle metric flag
    """

    def __init__(self, toggle_metrics: Optional[List[str]]=None):
        """
        Inits the callback
        :param toggle_metrics: A list of metrics to toggle (can be None), all metrics in the list need a "on" variable
                               that can be toggled
        """

        # init the base
        super().__init__()

        # set the metrics to toggle
        if toggle_metrics is None:
            self.metrics = []
        else:
            self.metrics = toggle_metrics

    def on_test_begin(self, logs):
        """
        A function called on test begin, toggles the metrics on
        :param logs: The logs
        """

        for metric in self.model.metrics:
            for custom_metric in self.metrics:
                if custom_metric in metric.name:
                    metric.on.assign(True)

    def on_test_end(self,  logs):
        """
        A function called at the end of the test, toggles the metrics of
        :param logs: The logs
        """

        for metric in self.model.metrics:
            for custom_metric in self.metrics:
                if custom_metric in metric.name:
                    metric.on.assign(False)


class ROIAccuracy(tf.keras.metrics.Metric):
    """
    Calculates the accuracy within the regions of interest
    Note: This metric can be toggled
    """

    def __init__(self, on_start=True, **kwargs):
        """
        Inits the metric
        :param on_start: Wheter the metric is turned on on start
        :param kwargs: keyword arguments forwarded to the base class init
        """

        # base class init
        super().__init__(**kwargs)
        self.tp = self.add_weight(name=f'true_positive', initializer='zeros')
        self.total_roi = self.add_weight(name=f'total_roi', initializer='zeros')
        self.on = tf.Variable(on_start)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric
        :param y_true: The label of the batch
        :param y_pred: The predictions of the batch
        :param sample_weight: Sample weights
        """
        # Use conditional to determine if computation is done
        if self.on:
            tp = tf.reduce_sum(tf.cast(y_pred > 0.5, tf.int32)*tf.cast(y_true > 0.5, tf.int32))
            self.tp.assign_add(tf.cast(tp, tf.float32))
            total_roi = tf.reduce_sum(tf.cast(y_true > 0.5, tf.int32))
            self.total_roi.assign_add(tf.cast(total_roi, tf.float32))

    def result(self):
        """
        Calculates the current result of the metric
        :return: The current result
        """
        return self.tp/self.total_roi

    def reset_state(self):
        """
        Resets the metric to default state
        """
        self.tp.assign(0.)
        self.total_roi.assign(0.0)


class AveragePrecision(tf.keras.metrics.Metric):
    """
    This is a TF metric used for to calculate the average precision for a given threshold,
    note that this is quite slow compared to the other TF functions because it is not a graph function
    Note: This metric can be toggled
    """

    def __init__(self, threshold, name=None, on_start=False, **kwargs):
        """
        Inits the metric
        :param threshold: The threshold used fort the calculation
        :param name: The name of the metrics to display
        :param on_start: Wheter the metric is turned on on start
        :param kwargs: Keyword arguments forwarded to the metric base class
        """
        # Initialise as normal and add flag variable for when to run computation
        if name is None:
            name = f"average_precision_{threshold}"
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name=f'true_positive', initializer='zeros')
        self.fp = self.add_weight(name=f'false_positive', initializer='zeros')
        self.fn = self.add_weight(name=f'false_negative', initializer='zeros')
        self.on = tf.Variable(on_start)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Updates the state of the metric
        :param y_true: The label of the batch
        :param y_pred: The predictions of the batch
        :param sample_weight: Sample weights
        """
        # Use conditional to determine if computation is done
        if self.on:
            # run computation
            ap, tp, fp, fn = tf.numpy_function(lambda true, pred, thres: metrics.average_precision(label(true),
                                                                                                   label(pred),
                                                                                                   thres),
                                               [tf.cast(y_true[...,0] > 0.5, tf.int32),
                                                tf.cast(y_pred[...,0] > 0.5, tf.int32),
                                                [self.threshold]],
                                               Tout=(tf.float32, tf.float32, tf.float32, tf.float32))
            self.tp.assign_add(tf.reduce_sum(tp))
            self.fp.assign_add(tf.reduce_sum(fp))
            self.fn.assign_add(tf.reduce_sum(fn))

    def result(self):
        """
        Calculates the current result of the metric
        :return: The current result
        """
        return self.tp/(self.tp + self.fp + self.fn)

    def reset_state(self):
        """
        Resets the metric to default state
        """
        self.tp.assign(0.)
        self.fp.assign(0.)
        self.fn.assign(0.)

