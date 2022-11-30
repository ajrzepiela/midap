import tensorflow as tf
from midap.networks import layers

def test_UNetLayerClassicDown():
    """
    Tests the UNetLayerClassicDown implementation
    """

    model = layers.UNetLayerClassicDown(filters=32, kernel_size=3,
                                        pool_size=(2, 2), dropout=0.5)

    # create an input
    inp = tf.ones((2, 4, 8, 16))
    out_conv, out_x = model(inp)

    # check shape
    assert out_conv.shape == (2, 4, 8, 32)
    assert out_x.shape == (2, 2, 4, 32)

def test_UNetLayerClassicUp():
    """
    Tests the UNetLayerClassicUp implementation
    """

    model = layers.UNetLayerClassicUp(filters=32, kernel_size=3, dropout=0.5)

    # create an input
    inp1 = tf.ones((2, 2, 4, 16))
    inp2 = tf.ones((2, 4, 8, 16))
    out = model(inp1, inp2)

    # check shape
    assert out.shape == (2, 4, 8, 32)
