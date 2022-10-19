import PySimpleGUI as sg
from datetime import datetime
import numpy as np
import git

# get all subclasses from the imcut
from midap.imcut import *
from midap.imcut import base_cutout

imcut_subclasses = [subclass.__name__ for subclass in base_cutout.CutoutImage.__subclasses__()]

# get all subclasses from the segmentations
from midap.segmentation import *
from midap.segmentation import base_segmentator

segmentation_subclasses = [subclass.__name__ for subclass in base_segmentator.SegmentationPredictor.__subclasses__()]

# set layout for GUI
sg.theme('LightGrey1')
appFont = ("Arial", 12)
sg.set_options(font=appFont)

# Seclection of Family Machine vs Well
layout = [[sg.Text('Data type', font='bold')],
          [sg.Radio('Family/Mother Machine', "RADIO1", default=True, key="family_machine"),
           sg.Radio('Well', "RADIO1", default=False, key="well")],
          [sg.Text('')],
          [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]

# Finalize layout
window = sg.Window('Parameters', layout).Finalize()

# read the values and close
event, values = window.read()
window.close()

# Throw an error if we pressed cancel or X
if event == 'Cancel' or event == None:
    exit(1)

# Common elements of the next GUI part
workflow = [[sg.Text('Part of pipeline', justification='center', size=(16, 1))],
            [sg.T("         "), sg.Radio('Segmentation and Tracking', "RADIO1", default=True, key="segm_track")],
            [sg.T("         "), sg.Radio('Segmentation', "RADIO1", default=False, key="segm_only")],
            [sg.T("         "), sg.Radio('Tracking', "RADIO1", default=False, key="track_only")]]

frames = [[sg.Text('Set frame number')],
          [sg.Input('0', size=(5, 30), key='start_frame'), sg.Text('-')],
          [sg.Input('100', size=(5, 30), key='end_frame')]]


# open file handle for settings.sh
with open("settings.sh", "w+") as file_settings:
    # write date time header
    current_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    file_settings.write(f"# Started run: {current_time}\n")

    # write the current git hash of the repo
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    file_settings.write(f"# Git hash: {sha}\n")

    # go through cases
    if values['family_machine'] == True:
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
            exit(1)

        # split the channel types
        channel_type_vals = [values['ch1']]
        # Only an emtpy string is False
        if values['ch2']:
            channel_type_vals += values['ch2'].split(",")

        # Read out the Radio Button
        run_options = ['BOTH', 'SEGMENTATION', 'TRACKING']
        cond_run = [values['segm_track'], values['segm_only'], values['track_only']]
        ix_cond = np.where(np.array(cond_run))[0][0]

        # Write the settings.sh
        file_settings.write(f"RUN_OPTION={run_options[ix_cond]}\n")
        if values['deconv'] == True:
            file_settings.write("DECONVOLUTION=deconv_family_machine\n")
        else:
            file_settings.write("DECONVOLUTION=no_deconv\n")
        file_settings.write(f"PHASE_SEGMENTATION={values['phase_segmentation']}\n")
        file_settings.write(f"START_FRAME={values['start_frame']}\n")
        file_settings.write(f"END_FRAME={values['end_frame']}\n")
        file_settings.write(f"DATA_TYPE=FAMILY_MACHINE\n")
        file_settings.write(f"PATH_FOLDER={values['folder_name']}/\n")
        file_settings.write(f"FILE_TYPE={values['file_type']}\n")
        file_settings.write(f"POS_IDENTIFIER={values['pos']}\n")
        file_settings.write(f"CHAMBER_CUTOUT={values['imcut']}\n")
        file_settings.write(f"SEGMENTATION_METHOD={values['seg_method']}\n")

        for i, s in enumerate(channel_type_vals):
            file_settings.write(f"CHANNEL_{i + 1}={s}\n")
        file_settings.write(f"NUM_CHANNEL_TYPES={len(channel_type_vals)}\n")

    elif values['well'] == True:
        # Well specific layout
        layout_well = [[sg.Frame('Conditional Run', [[
            sg.Column(workflow, background_color='white'),
            sg.Column(frames)
        ]])],
                       [sg.Text('Filename', key='title_file_name', font='bold')],
                       [sg.Column([[sg.Input(key='file_name'), sg.FileBrowse()]], key='col_file_name')],
                       [sg.Text('')],
                       [sg.Text('Filetype (e.g. tif, tiff, ...)', key='title_file_type', font='bold')],
                       [sg.Input(key='file_type')],
                       [sg.Text('Identifier of Position/Experiment (e.g. Pos, pos)', key='pos_id', font='bold')],
                       [sg.Input(key='pos')],
                       [sg.Text('')],
                       [sg.Text('Preprocessing', font='bold')],
                       [sg.Checkbox('Deconvolution of images', key='deconv', font='bold')],
                       [sg.Text('')],
                       [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]

        # Finalize the layout
        window = sg.Window('Parameters', layout_well).Finalize()

        # Read params and close window
        event, values = window.read()
        window.close()

        # Throw an error if we pressed cancel or X
        if event == 'Cancel' or event == None:
            exit(1)

        # Read out the radio button
        run_options = ['BOTH', 'SEGMENTATION', 'TRACKING']
        cond_run = [values['segm_track'], values['segm_only'], values['track_only']]
        ix_cond = np.where(np.array(cond_run))[0][0]

        # write the settings.sh
        file_settings.write(f"RUN_OPTION={run_options[ix_cond]}\n")
        if values['deconv'] == True:
            file_settings.write("DECONVOLUTION=deconv_well\n")
        else:
            file_settings.write("DECONVOLUTION=no_deconv\n")
        file_settings.write(f"START_FRAME={values['start_frame']}\n")
        file_settings.write(f"END_FRAME={values['end_frame']}\n")
        file_settings.write(f"DATA_TYPE=WELL\n")
        file_settings.write(f"PATH_FILE={values['file_name']}\n")
        file_settings.write(f"FILE_TYPE={values['file_type']}\n")
        file_settings.write(f"POS_IDENTIFIER={values['pos']}\n")
