import PySimpleGUI as sg
import numpy as np

from midap.config import Config

# Get all subclasses for the dropdown menus
###########################################

# get all subclasses from the imcut
from midap.imcut import *
from midap.imcut import base_cutout

imcut_subclasses = [subclass.__name__ for subclass in base_cutout.CutoutImage.__subclasses__()]

# get all subclasses from the segmentations
from midap.segmentation import *
from midap.segmentation import base_segmentator

segmentation_subclasses = [subclass.__name__ for subclass in base_segmentator.SegmentationPredictor.__subclasses__()]

# get all subclasses from the tracking
from midap.tracking import *
from midap.tracking import base_tracking

tracking_subclasses = [subclass.__name__ for subclass in base_tracking.Tracking.__subclasses__()]

# main function of the App
##########################

def main(config_file="settings.ini"):
    """
    The main function of the App
    :param config_file: Name of the config file to save
    """
    # set layout for GUI
    sg.theme('LightGrey1')
    appFont = ("Arial", 12)
    sg.set_options(font=appFont)

    # Common elements of the next GUI part
    workflow = [[sg.Text('Part of pipeline', justification='center', size=(16, 1))],
                [sg.T("         "), sg.Radio('Segmentation and Tracking', "RADIO1", default=True, key="segm_track")],
                [sg.T("         "), sg.Radio('Segmentation', "RADIO1", default=False, key="segm_only")],
                [sg.T("         "), sg.Radio('Tracking', "RADIO1", default=False, key="track_only")]]

    frames = [[sg.Text('Set frame number')],
              [sg.Input('0', size=(5, 30), key='start_frame'), sg.Text('-')],
              [sg.Input('100', size=(5, 30), key='end_frame')]]


    # Family Machine specific layout
    layout_family_machine = [[sg.Frame('Conditional Run', [[
        sg.Column(workflow, background_color='white'),
        sg.Column(frames)
    ]])],
                             [sg.Text('Foldername', key='title_folder_name', font='bold')],
                             [sg.Column([[sg.Input(key='folder_name'), sg.FolderBrowse()]], key='col_folder_name')],
                             [sg.Text('')],
                             [sg.Text('Filetype (e.g. tif, tiff, ...)', key='title_file_type', font='bold')],
                             [sg.Input(key='file_type')],
                             [sg.Text('Identifier of Position/Experiment (e.g. Pos, pos)', key='pos_id',
                                      font='bold')],
                             [sg.Input(key='pos')],
                             [sg.Text('Identifier of phase channel (e.g. Phase, PH, ...)', key='phase_check',
                                      font='bold')],
                             [sg.Input(key='ch1'),
                              sg.Checkbox('Segmentation/Tracking', key='phase_segmentation', font='bold')],
                             [sg.Text('Comma separated list of identifiers of additional \n'
                                      'channels (e.g. eGFP,GFP,YFP,mCheery,TXRED, ...)',
                                      key='channel_1', font='bold')],
                             [sg.Input(key='ch2')],
                             [sg.Text('Select how the chamber cutout should be performed: ',
                                      key='imcut_text', font='bold')],
                             [sg.DropDown(key='imcut', values=imcut_subclasses, default_value="InteractiveCutout")],
                             [sg.Text('Select how the cell segmentation should be performed: ',
                                      key='seg_method_text', font='bold')],
                             [sg.DropDown(key='seg_method', values=segmentation_subclasses,
                                          default_value="UNetSegmentation")],
                             [sg.Text('Select how the cell tracking should be performed: ',
                                      key='track_method_text', font='bold')],
                             [sg.DropDown(key='track_method', values=tracking_subclasses,
                                          default_value="DeltaV2Tracking")],
                             [sg.Text('')],
                             [sg.Text('Preprocessing', font='bold')],
                             [sg.Checkbox('Deconvolution of images', key='deconv', font='bold')],
                             [sg.Text('')],
                             [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]

    # Finalize the layout
    window = sg.Window('Parameters', layout_family_machine).Finalize()

    # Read Params and close window
    event, values = window.read()
    window.close()

    # Throw an error if we pressed cancel or X
    if event == 'Cancel' or event == None:
        # TODO: Think about what should happen here
        exit(1)

    # Set the general parameter
    general = {}

    # get all the channels
    channels = values['ch1']
    # Only an emtpy string is False
    if values['ch2']:
        channels += f",{values['ch2']}"
    general["Channels"] = channels

    # Read out the Radio Button
    run_options = ['both', 'segmentation', 'tracking']
    cond_run = [values['segm_track'], values['segm_only'], values['track_only']]
    ix_cond = np.where(np.array(cond_run))[0][0]
    general["RunOption"] = run_options[ix_cond]

    # deconv
    if values['deconv'] == True:
        general["Deconvolution"] = "deconv_family_machine"
    else:
        general["Deconvolution"] = "no_deconv"

    # The remaining generals
    general["StartFrame"] = values['start_frame']
    general["EndFrame"] = values['end_frame']
    general["DataType"] = "Family_Machine"
    general["FolderPath"] = values['folder_name']
    general["PosIdentifier"] = values['pos']
    general["FileType"] = values['file_type']
    general["PhaseSegmentation"] = values['phase_segmentation']

    # the image cut method
    cut_img = {"Class": values['imcut']}

    # the segmentation method
    segmentation = {"Class": values['seg_method']}

    # the tracking method
    tracking = {"Class": values['track_method']}

    # init the config
    config = Config(fname=config_file, general=general, segmentation=segmentation, cut_img=cut_img, tracking=tracking)

    # write to file
    config.to_file(config_file, overwrite=True)

# Run as script
###############

if __name__ == "__main__":
    main()
