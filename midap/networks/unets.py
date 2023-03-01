from typing import Optional, List

import tensorflow as tf


class UNetBaseClass(tf.keras.Model):

    """
    This is a generic implementation of a UNet with various loss functions etc.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the Class as tf.keras.Model
        :param args: arguments that are forwarded to the tf.keras.Model init
        :param kwargs: keyword arguments that are forwarded to the tf.keras.Model init
        """

        # This line is mandatory for all TF model subclasses
        super().__init__(*args, **kwargs)

    @classmethod
    def convert_to_logits(cls, y_pred: tf.Tensor):
        """
        Transforms predictions to log-odds
        :param y_pred: The un-normalized predictions
        :return: log-odds calculated with backend epsilon clipped predictions to avoid under and over-flow
        """

        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.math.log(y_pred / (1 - y_pred))

    def pos_weighted_binary_crossentropy(self, y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor):
        """
        Weighted binary cross entropy a la tf.nn.weighted_cross_entropy_with_logits
        :param y_true: The ground truth
        :param y_pred: The predictions
        :param weights: The weights for the loss
        :return: The calculated loss
        """

        # convert to logit and calculate loss
        y_pred = self.convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=weights)

        return tf.reduce_mean(loss)

    def balanced_binary_crossentropy(self, y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor):
        """
        Weighted binary cross entropy a la tf.nn.weighted_cross_entropy_with_logits where the supplied weights
        are adapted according to weights/(1 - weights) and the final loss is calculated with a weighted average
        :param y_true: The ground truth
        :param y_pred: The predictions
        :param weights: The weights for the loss
        :return: The calculated loss
        """
        # adapt weights
        pos_weights = weights/(1.0 - weights)

        # convert to logit and calculate loss
        y_pred = self.convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weights)

        return tf.reduce_mean(loss*weights)

    def weighted_binary_crossentropy(self, y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor):
        """
        A loss that implements the weighted binary cross entropy
        :param y_true: The ground truth
        :param y_pred: The predictions
        :param weights: The weights for the loss
        :return: The calculated binary cross entropy
        """
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        return bce(y_true, y_pred, sample_weight=weights)

    def weighted_categorical_crossentropy(self, y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        :param y_true: The ground truth
        :param y_pred: The predictions
        :param weights: The weights for the loss (one entry per class)
        :return: The calculated binary cross entropy
        """

        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.math.log(y_pred) * weights
        loss = -tf.math.reduce_sum(loss, -1)
        return loss

    def focal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, alpha=0.25, gamma=2.0):
        """
        Implements the focal loss with alpha and gamma parameters see e.g.
        https://paperswithcode.com/method/focal-loss
        :param y_true: The ground truth
        :param y_pred: The predictions
        :param alpha: alpha parameter of the focal loss
        :param gamma: gamma parameter of the focal loss
        :return: the calculated loss
        """

        # convert to logits
        logits = self.convert_to_logits(y_pred=y_pred)

        # get the weights
        weight_a = alpha * (1 - y_pred) ** gamma * y_true
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - y_true)

        # claculate the loss
        loss = (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

        return tf.reduce_mean(loss)

class UNetv1(UNetBaseClass):

    """
    This implements the standard class of UNets used in the pipeline
    """

    def __init__(self, input_size=(256, 512, 1), dropout=0.5, inference=False, metrics: Optional[List]=None):
        """
        Initializes the UNet
        :param input_size: Size of the input
        :param dropout: Dropout factor for the dropout layers
        :param inference: If False, model is compiled for training
        :param metrics: Additional metric to add for the training (only if inference=False)
        """

        # define the layers
        inp = tf.keras.layers.Input(input_size)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inp)
        conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = tf.keras.layers.Dropout(dropout)(conv4)
        pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(drop4)

        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = tf.keras.layers.Dropout(dropout)(conv5)

        up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
        merge6 = tf.keras.layers.Concatenate(axis=-1)([conv4, up6])
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same',
                                     kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
        merge7 = tf.keras.layers.Concatenate(axis=-1)([conv3, up7])
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same',
                                     kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
        merge8 = tf.keras.layers.Concatenate(axis=-1)([conv2, up8])
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same',
                                     kernel_initializer='he_normal')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
        merge9 = tf.keras.layers.Concatenate(axis=-1)([conv1, up9])
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

        # do the super init depending on inference or not
        if inference:
            super().__init__(inputs=inp, outputs=conv10)
        else:
            # addtional weight tensor for the lass
            weights_tensor = tf.keras.layers.Input(input_size)
            targets_tensor = tf.keras.layers.Input(input_size)
            super().__init__(inputs=[inp, weights_tensor, targets_tensor], outputs=conv10)

            # now we compile the model
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

            # We add the loss with the add_loss method because keras input layers are no longer allowed in
            # loss functions
            self.add_loss(self.weighted_binary_crossentropy(y_true=targets_tensor, y_pred=conv10,
                                                            weights=weights_tensor))
            if metrics is not None:
                metrics = ['accuracy'] + metrics
            else:
                metrics = ['accuracy']
            self.compile(optimizer=self.optimizer, loss=None, metrics=metrics)
