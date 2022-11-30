import numpy as np
from midap.networks import deltav2
from skimage.measure import label
from pathlib import Path

def test_unet_track():
    """
    Tests the unet_track routine of deltav2
    """

    # weight path
    weight_path = Path(__file__).absolute().parent.parent.parent
    weight_path = weight_path.joinpath("model_weights", "model_weights_tracking", "unet_pads_track.hdf5")

    # load the model
    model = deltav2.unet_track(pretrained_weights=weight_path, input_size=(128, 128, 4))

    # we generate fake input, the channels are original image, seg of original (only relevant cell),
    # next frame original, next frame seg (full)
    inp = np.zeros((1, 128, 128, 4))

    # we create a large rectangular cell
    inp[:,25:100,75:100,:] = 1

    # call the model
    out = model(inp).numpy()

    # label the parts, 0.9 is used in the clean_cur_frame method of the base tracking class
    _, num = label((out[0] > 0.9).astype(int), return_num=True)

    # check that there is no daughter cell
    assert num == 1

    # now we create a split event
    inp[:, 50:55, 75:100, 2:] = 0

    # call the model
    out = model(inp).numpy()

    # label the parts, 0.9 is used in the clean_cur_frame method of the base tracking class
    _, num = label((out[0] > 0.9).astype(int), return_num=True)

    # check that there is a daughter cell
    assert num == 2
