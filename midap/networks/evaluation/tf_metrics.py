import tensorflow as tf
from cellpose import metrics
from typing import List, Optional

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

class AveragePrecision(tf.keras.metrics.Metrics):
    """
    This is a TF metric used for to calculate the average precision
    """

    def __init__(self, **kwargs):
        # Initialise as normal and add flag variable for when to run computation
        super(MyCustomMetric, self).__init__(**kwargs)
        self.metric_variable = self.add_weight(name='metric_varaible', initializer='zeros')
        self.on = tf.Variable(False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Use conditional to determine if computation is done
        if self.on:
            # run computation
            self.metric_variable.assign_add(computation_result)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)

