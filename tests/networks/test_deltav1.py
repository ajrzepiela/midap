import numpy as np
from midap.networks import deltav1
from pathlib import Path

def test_unet_track():
    """
    Tests the unet_track routine of deltav1
    """

    # get the model
    model = deltav1.unet_track(input_size=(128, 128, 4))

    # load the weights
    weight_path = Path(__file__).absolute().parent.parent.parent
    weight_path = weight_path.joinpath("model_weights", "model_weights_tracking", "unet_moma_track_multisets.hdf5")
    model.load_weights(weight_path)

    # we generate fake input, the channels are original image, seg of original (only relevant cell),
    # next frame original, next frame seg (full)
    inp = np.zeros((1, 128, 128, 4))

    # we create a large rectangular cell
    inp[:,25:100,75:100,:] = 1

    # call the model
    out = model(inp).numpy()

    # check that there is no daughter cell
    assert np.any(out[0,:,:,0] > 0.5) and np.all(out[0,:,:,1] < 0.5)

    # now we create a split event
    inp[:, 50:55, 75:100, 2:] = 0

    # call the model
    out = model(inp).numpy()

    # check that there is a daughter cell
    assert np.any(out[0,:,:,1] > 0.5) and np.any(out[0,:,:,1] > 0.5)
