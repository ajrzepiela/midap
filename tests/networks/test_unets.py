import tensorflow as tf
import numpy as np
from midap.networks import unets
from pathlib import Path

def test_UNetv1():
    """
    Tests the UNetv1 class
    """

    # path to weights
    weight_path = Path(__file__).absolute().parent.parent.parent
    weight_path = weight_path.joinpath("model_weights", "model_weights_legacy",
                                       "model_weights_Caulobacter-crescentus-CB15_mKate2_v01.h5")

    # create the model (no inference mode)
    inp_size = (64, 128, 1)
    inp = tf.ones((2, ) + inp_size)
    model1 = unets.UNetv1(input_size=inp_size, inference=False)
    model1.load_weights(weight_path)

    # call we need inp, weight tensor and targets
    out1 = model1((inp, inp, inp)).numpy()

    # check shape
    assert out1.shape == inp.shape

    # load with inference on
    model1 = unets.UNetv1(input_size=inp_size, inference=True)
    model1.load_weights(weight_path)

    # call we need only inp
    out2 = model1(inp).numpy()

    # check that we have the same output
    assert np.allclose(out1, out2)
