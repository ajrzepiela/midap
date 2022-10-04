import argparse

import sys
sys.path.append('../src')
import os
import re

from unet_prediction import SegmentationPredictor

### Functions
#############

def save_weights(channel, weight_path):
    """
    Saves the path to the model weights into the settings file
    :param channel: The channel of the weights selected
    :param weight_path: Path to the weights
    """
    # Save the selected weights
    with open("settings.sh", "r+") as file_settings:
        # read the file content
        content =file_settings.read()

        # if there is no variable -> write new
        if f"MODEL_WEIGHTS_{channel}" in content:
            # we replace the path
            content = re.sub(f"MODEL_WEIGHTS_{channel}\=.*",
                             f"MODEL_WEIGHTS_{channel}={weight_path}",
                             content)
            # truncate, set stream to start and write
            file_settings.truncate(0)
            file_settings.seek(0)
            file_settings.write(content)
        else:
            # we write a new variable
            file_settings.write(f"MODEL_WEIGHTS_{args.path_channel}={os.path.abspath(pred.model_weights)}\n")

# Main
######

if __name__ == "__main__":

    # arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_model_weights", type=str, required=True, help="Path to the model weights that will be used "
                                                                              "for the segmentation.")
    parser.add_argument("--path_pos", type=str, required=True, help="Path to the current identifier folder to work on.")
    parser.add_argument("--path_channel", type=str, required=True, help="Name of the current channel to process.")
    parser.add_argument("--batch_mode", action="store_true", help="Flag for batch mode.")
    parser.add_argument("--postprocessing", action="store_true", help="Flag for postprocessing.")
    args = parser.parse_args()

    # get the Predictor
    pred = SegmentationPredictor(path_model_weights=args.path_model_weights, postprocessing=args.postprocessing)

    # set the paths
    path_cut = os.path.join(args.path_channel, "cut_im")
    path_seg = os.path.join(args.path_channel, "seg_im")
    path_seg_track = os.path.join(args.path_channel, "input_ilastik_tracking")

    # We use the same weight for all channels
    path_model_weights = None
    if args.batch_mode:
        # Readout the parameters
        with open("settings.sh","r") as file_settings:
            lines = file_settings.readlines()

        # Transform to dict, ignore comments in file
        list_items = [l.replace('\n', '').split('=') for l in lines if not l.startswith("#")]
        params_dict = {l[0]: l[1] for l in list_items}

        # check if the path for the weights is set in
        param = f'MODEL_WEIGHTS_{args.path_channel}'
        if param in params_dict and args.path_channel != params_dict['CHANNEL_1']:
            # we set the path for the weights
            path_model_weights = params_dict[param]

            # we already have the weights set -> run the
            pred.run_image_stack(args.path_pos, path_cut, path_seg, path_seg_track, path_model_weights)

    # Select the weights if not set by the batch mode
    if path_model_weights is None:
        pred.select_weights(args.path_pos, path_cut, path_seg)
        path_model_weights = os.path.abspath(pred.model_weights)

    # Save the selected weights
    save_weights(args.path_channel, path_model_weights)

    # run the stack
    pred.run_image_stack(args.path_pos, path_cut, path_seg, path_seg_track, path_model_weights)
