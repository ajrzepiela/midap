import tensorflow as tf
from typing import Union

class UNetLayerClassicDown(tf.keras.layers.Layer):

    """
    This class implements a basic Unet block
    """

    def __init__(self, filters: int, kernel_size: Union[int, tuple], activation='relu', padding='same', dropout=None,
                 kernel_initializer='he_normal', pool_size=None):
        """
        Initializes the layer
        :param filters: The number of filters for the convolution
        :param kernel_size: The kernel size of the convolutions, can be either a single int (square kernel) or a tuple
        :param activation: The activation function as string, must be a member of tf.keras.activations
        :param padding: The padding type
        :param dropout: The dropout rate applied after the two convolutions, defaults to no dropout
        :param kernel_initializer: The kernel initialization method
        :param pool_size: The pooling window applied after the dropout, defaults to no pooling (same size output)
        """

        # This line is mandatory for all Layer subclasses
        super(UNetLayerClassicDown, self).__init__()

        # convolutions
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, activation=activation, kernel_size=kernel_size,
                                            padding=padding, kernel_initializer=kernel_initializer)
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, activation=activation, kernel_size=kernel_size,
                                            padding=padding, kernel_initializer=kernel_initializer)

        # dropout and pooling
        if dropout is not None:
            self.dropout = tf.keras.layers.Dropout(rate=dropout)
        else:
            self.dropout = tf.keras.activations.linear
        if pool_size is not None:
            self.pool = tf.keras.layers.MaxPool2D(pool_size=pool_size)
        else:
            self.pool = tf.keras.activations.linear

    def call(self, input: tf.Tensor):
        """
        Calls the UNetLayer
        :param input: The input to the layer
        :return: The output of the convolution for the skip connection and the final output (after dropout and pooling)
        """

        # the rest of the layer
        x = self.conv1(input)
        conv_out = self.conv2(x)
        x = self.dropout(conv_out)
        x = self.pool(x)

        # Return the output of the convolutions for the skip connections and x
        return conv_out, x


class UNetLayerClassicUp(tf.keras.layers.Layer):

    """
    This class implements a basic Unet block
    """

    def __init__(self, filters: int, kernel_size: Union[int, tuple], activation='relu', padding='same', dropout=None,
                 kernel_initializer='he_normal', upsize=(2,2)):
        """
        Initializes the layer
        :param filters: The number of filters for the convolution
        :param kernel_size: The kernel size of the convolutions, can be either a single int (square kernel) or a tuple
        :param activation: The activation function as string, must be a member of tf.keras.activations
        :param padding: The padding type
        :param dropout: The dropout rate applied after the two convolutions, defaults to no dropout
        :param kernel_initializer: The kernel initialization method
        :param upsize: The kernel size to use for the upsampling
        """

        # This line is mandatory for all Layer subclasses
        super(UNetLayerClassicUp, self).__init__()

        # up smpling
        self.upsampling = tf.keras.layers.UpSampling2D(size=upsize)
        self.conv0 = tf.keras.layers.Conv2D(filters=filters, activation=activation, kernel_size=2,
                                            padding=padding, kernel_initializer=kernel_initializer)
        # convolutions
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, activation=activation, kernel_size=kernel_size,
                                            padding=padding, kernel_initializer=kernel_initializer)
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, activation=activation, kernel_size=kernel_size,
                                            padding=padding, kernel_initializer=kernel_initializer)

        # dropout and pooling
        if dropout is not None:
            self.dropout = tf.keras.layers.Dropout(rate=dropout)
        else:
            self.dropout = tf.keras.activations.linear

    def call(self, input1: tf.Tensor, input2: tf.Tensor):
        """
        Calls the UNetLayer
        :param input1: The input to the layer
        :param input2: Additional input to the layer, used for the skip connections
        :return: The output of the called layer
        """

        # We start with the upsampling
        x = self.upsampling(input1)
        x = self.conv0(x)

        # merge the second input if necessary
        x = tf.concat([input2, x], axis=-1)

        # the rest of the layer
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)

        return x