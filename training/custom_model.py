from typing import Optional, List

import tensorflow as tf

from midap.networks.layers import UNetLayerClassicDown, UNetLayerClassicUp
from midap.networks.unets import UNetBaseClass


class CustomUNet(UNetBaseClass):

    """
    This class implements a custom UNet based on the UNetBaseClass with it's preimplemented functions
    """

    def __init__(
        self,
        input_size=(256, 512, 1),
        dropout=0.5,
        inference=False,
        metrics: Optional[List] = None,
    ):
        """
        Initializes the UNet
        :param input_size: Size of the input
        :param dropout: Dropout factor for the dropout layers
        :param inference: If False, model is compiled for training
        """

        # The input layer
        self.inp = tf.keras.layers.Input(input_size)

        # We go down the UNet
        conv1, down1 = UNetLayerClassicDown(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            dropout=None,
            kernel_initializer="he_normal",
            pool_size=(2, 2),
        )(self.inp)
        conv2, down2 = UNetLayerClassicDown(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same",
            dropout=None,
            kernel_initializer="he_normal",
            pool_size=(2, 2),
        )(down1)
        conv3, down3 = UNetLayerClassicDown(
            filters=256,
            kernel_size=3,
            activation="relu",
            padding="same",
            dropout=dropout,
            kernel_initializer="he_normal",
            pool_size=(2, 2),
        )(down2)
        conv4, down4 = UNetLayerClassicDown(
            filters=512,
            kernel_size=3,
            activation="relu",
            padding="same",
            dropout=dropout,
            kernel_initializer="he_normal",
            pool_size=(2, 2),
        )(down3)
        conv5, down5 = UNetLayerClassicDown(
            filters=1024,
            kernel_size=3,
            activation="relu",
            padding="same",
            dropout=dropout,
            kernel_initializer="he_normal",
            pool_size=None,
        )(down4)

        # and up again (with skip connections)
        up1 = UNetLayerClassicUp(
            filters=512,
            kernel_size=3,
            activation="relu",
            padding="same",
            dropout=None,
            kernel_initializer="he_normal",
        )(down5, conv4)
        up2 = UNetLayerClassicUp(
            filters=256,
            kernel_size=3,
            activation="relu",
            padding="same",
            dropout=None,
            kernel_initializer="he_normal",
        )(up1, conv3)
        up3 = UNetLayerClassicUp(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same",
            dropout=None,
            kernel_initializer="he_normal",
        )(up2, conv2)
        up4 = UNetLayerClassicUp(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            dropout=None,
            kernel_initializer="he_normal",
        )(up3, conv1)

        # output layer
        self.out = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(up4)

        # compile the model
        if inference:
            super().__init__(inputs=self.inp, outputs=self.out)
        else:
            # addtional weight tensor for the lass
            weights_tensor = tf.keras.layers.Input(input_size)
            targets_tensor = tf.keras.layers.Input(input_size)
            super().__init__(
                inputs=[self.inp, weights_tensor, targets_tensor], outputs=self.out
            )

            # now we compile the model
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

            # We add the loss with the add_loss method because keras input layers are no longer allowed in
            # loss functions
            self.add_loss(
                self.weighted_binary_crossentropy(
                    y_true=targets_tensor, y_pred=self.out, weights=weights_tensor
                )
            )
            if metrics is not None:
                metrics = ["accuracy"] + metrics
            else:
                metrics = ["accuracy"]
            self.compile(optimizer=self.optimizer, loss=None, metrics=metrics)

    def save(self, filepath, **kwargs):
        """
        Saves the model in inference mode as a TF model
        :param filepath: Path to save directory
        :param kwargs: Additional keyword args forwarded to tf.keras.models.save_model
        """

        model = tf.keras.Model(inputs=self.inp, outputs=self.out)
        model.compile(optimizer=self.optimizer)
        model.save(filepath=filepath, **kwargs)
