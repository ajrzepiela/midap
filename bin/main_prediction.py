import argparse

import sys
sys.path.append('../src')
import os

from unet_prediction import SegmentationPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--path_model_weights")
parser.add_argument("--path_pos")
parser.add_argument("--path_channel")
parser.add_argument("--postprocessing")
parser.add_argument("--batch_mode")
args = parser.parse_args()

pred = SegmentationPredictor(path_model_weights=args.path_model_weights, postprocessing=bool(int(args.postprocessing)))
path_cut = f"/{args.path_channel}/cut_im/"
path_seg = f"/{args.path_channel}/seg_im/"
path_seg_track = f"/{args.path_channel}/input_ilastik_tracking/"


if bool(int(args.batch_mode)) == False:
    pred.select_weights(args.path_pos, path_cut, path_seg)

    # Save the selected weights
    with open("settings.sh", "a") as file_settings:
        file_settings.write(f"MODEL_WEIGHTS_{args.path_channel}={os.path.abspath(pred.model_weights)}\n")

    pred.run_image_stack(args.path_pos, path_cut, path_seg, path_seg_track, pred.model_weights)

elif bool(int(args.batch_mode)) == True:
    file_settings = open("settings.sh","r")


    list_params= file_settings.readlines()

    list_items = [l.replace('\n', '').split('=') for l in list_params]
    keys = [l[0] for l in list_items]
    values = [l[1] for l in list_items]
    params_dict = dict()

    for k, v in zip(keys, values):
        params_dict[k]=v

    file_settings.close()
    param = 'MODEL_WEIGHTS_'+args.path_channel

    if any([param in item for item in list(params_dict.keys())]):
        if args.path_channel==params_dict['CHANNEL_2']:
            path_model_weights = params_dict['MODEL_WEIGHTS_' + params_dict['CHANNEL_2']]
        elif args.path_channel==params_dict['CHANNEL_3']:
            path_model_weights = params_dict['MODEL_WEIGHTS_' + params_dict['CHANNEL_3']]
        pred.run_image_stack(args.path_pos, path_cut, path_seg, path_seg_track, path_model_weights)
    elif not any([param in item for item in list(params_dict.keys())]):
        pred.select_weights(args.path_pos, path_cut, path_seg)

        file_settings = open("settings.sh","a") 


        file_settings.write("MODEL_WEIGHTS_" + args.path_channel + "=" + pred.model_weights + "\n")
        file_settings.close()
        pred.run_image_stack(args.path_pos, path_cut, path_seg, path_seg_track, pred.model_weights)

