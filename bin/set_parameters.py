import PySimpleGUI as sg
import yaml
import numpy as np
import os

# set layout for GUI
sg.theme('LightGrey1')
appFont = ("Arial", 12)
sg.set_options(font=appFont)

def Button(*args, **kwargs):
    return sg.Col([[sg.Button(*args, **kwargs)]], pad=(0,0))

layout = [[sg.Text('Data type', font='bold')],
          [sg.Checkbox('Family/Mother Machine', key='family_machine', default=False),
           sg.Checkbox('Well', key='well', default=False)],
	        [sg.Text('')],
          [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]

# ------ Column Definition ------ #
column1 = [[sg.Text('Part of pipeline', justification='center', size=(12, 1))],
           [sg.T("         "), sg.Radio('Segmentation and Tracking', "RADIO1", default=True, key="segm_track")],
           [sg.T("         "), sg.Radio('Segmentation', "RADIO1", default=False, key="segm_only")],
           [sg.T("         "), sg.Radio('Tracking', "RADIO1", default=False, key="track_only")]]

layout_family_machine = [[sg.Frame('Conditional Run',[[
                          sg.Column(column1, background_color='white'),
                          sg.Column(
                          [[sg.Text('Set frame number')],
                          [sg.Input('0', size=(5, 30), key='start_frame'), sg.Text('-')],
                          [sg.Input('100', size=(5, 30), key='end_frame')]])
                          ]])],
                         [sg.Text('Foldername', key = 'title_folder_name', font='bold')],
                         [sg.Column([[sg.Input(key='folder_name'), sg.FolderBrowse()]], key='col_folder_name')],
                         [sg.Text('')],
                         [sg.Text('Filetype (e.g. tif, tiff, ...)', key = 'title_file_type', font='bold')],
                         [sg.Input(key='file_type')],
                         [sg.Text('Identifier of phase channel (e.g. Phase, PH, ...)', key =  'phase_check', font='bold')],
                         [sg.Input(key='ch1'), sg.Checkbox('Segmentation', key='phase_segmentation', font='bold')],
                         [sg.Text('Identifier of additional channel type 1 (e.g. eGFP, GFP, YFP, ...)', key = 'channel_1', font='bold')],
                         [sg.Input(key='ch2')],
                         [sg.Text('Identifier of additional channel type 2 (e.g. mCheery, TXRED, ...)', key = 'channel_2', font='bold')],
                         [sg.Input(key='ch3')],
                         [sg.Text('')],
                         [sg.Text('Preprocessing', font = 'bold')],
                         [sg.Checkbox('Deconvolution of images', key='deconv', font='bold')],
                         [sg.Text('')],
                         [sg.Text('Path to Matlab root folder', font='bold')],
                         [sg.Input(key='matlab_root')],
                         [sg.Text('')],
                         [sg.Text('SuperSegger constants', font='bold')],
                         [sg.Text('Constants'), sg.Input('100XPa', key='constants')],
                         [sg.Text('Time Step'), sg.Input('1', key='time_step')],
                         [sg.Text('Neighbor Flag'), sg.Input('true', key='neighbor_flag')],
                         [sg.Text('Minimal cell age'), sg.Input('3', key='min_cell_age')],
	                     [sg.Text('')],
                         [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]

column2 = [[sg.Text('Part of pipeline', justification='center', size=(12, 1))],
           [sg.T("         "), sg.Radio('Segmentation and Tracking', "RADIO1", default=True, key="segm_track")],
           [sg.T("         "), sg.Radio('Segmentation', "RADIO1", default=False, key="segm_only")],
           [sg.T("         "), sg.Radio('Tracking', "RADIO1", default=False, key="track_only")]]
layout_well = [[sg.Frame('Conditional Run',[[
                          sg.Column(column2, background_color='white'),
                          sg.Column(
                          [[sg.Text('Set frame number')],
                          [sg.Input('0', size=(5, 30), key='start_frame'), sg.Text('-')],
                          [sg.Input('100', size=(5, 30), key='end_frame')]])
                          ]])],
               [sg.Text('Filename', key = 'title_file_name', font='bold')],
               [sg.Column([[sg.Input(key='file_name'), sg.FileBrowse()]], key='col_file_name')],
               [sg.Text('')],
               [sg.Text('Filetype (e.g. tif, tiff, ...)', key = 'title_file_type', font='bold')],
               [sg.Input(key='file_type')],
               [sg.Text('')],
               [sg.Text('Path to Matlab root folder', font='bold')],
               [sg.Input(key='matlab_root')],
               [sg.Text('SuperSegger constants', font='bold')],
               [sg.Text('Constants')],
               [sg.Input('100XPa', key='constants')],
               [sg.Text('Time Step')],
               [sg.Input('1', key='time_step')],
               [sg.Text('Neighbor Flag')],
               [sg.Input('true', key='neighbor_flag')],
               [sg.Text('Minimal cell age')],
               [sg.Input('1', key='min_cell_age')],
	           [sg.Text('')],
               [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]

window = sg.Window('Parameters', layout).Finalize()

event, values = window.read()
window.close()

if values['family_machine'] == True:
    window = sg.Window('Parameters', layout_family_machine).Finalize()

    event, values = window.read()

    window.close()

    channel_type_vals = [values['ch1'], values['ch2'], values['ch3']]
    sel_channel_types = [x for x in channel_type_vals if x]

    run_options = ['BOTH', 'SEGMENTATION', 'TRACKING']
    cond_run = [values['segm_track'], values['segm_only'], values['track_only']]
    ix_cond = np.where(np.array(cond_run))[0][0]

    dict_file = [{'FOLDERNAME' : values['folder_name']},
    {'CHANNELS' : sel_channel_types}]

    file_settings = open("settings.sh","w") 
    file_settings.write("RUN_OPTION=" + run_options[ix_cond] + "\n")
    file_settings.write("DECONVOLUTION=" + str(bool(values['deconv'])) + "\n")
    file_settings.write("PHASE_SEGMENTATION=" + str(bool(values['phase_segmentation'])) + "\n")
    file_settings.write("START_FRAME=" + values['start_frame'] + "\n") 
    file_settings.write("END_FRAME=" + values['end_frame'] + "\n") 
    file_settings.write("DATA_TYPE=FAMILY_MACHINE" + "\n") 
    file_settings.write("PATH_FOLDER=" + values['folder_name'] + "/ \n") 
    file_settings.write("FILE_TYPE=" + values['file_type'] + "\n") 
    file_settings.write("MATLAB_ROOT=" + values['matlab_root'] + "\n") 
    file_settings.write("CONSTANTS=" + values['constants'] + "\n") 
    file_settings.write("TIME_STEP=" + values['time_step'] + "\n") 
    file_settings.write("NEIGHBOR_FLAG=" + str(int(bool(values['neighbor_flag']))) + "\n") 
    file_settings.write("MIN_CELL_AGE=" + values['min_cell_age'] + "\n") 

    # for i, s in enumerate(sel_cell_types):
    #     file_settings.write("CELL_TYPE_" + str(i + 1) + "=" + s + "\n")
    # file_settings.write("NUM_CELL_TYPES=" + str(len(sel_cell_types)) + "\n")

    for i, s in enumerate(sel_channel_types):
            file_settings.write("CHANNEL_" + str(i + 1) + "=" + s + "\n")
    file_settings.write("NUM_CHANNEL_TYPES=" + str(len(sel_channel_types)) + "\n")
    file_settings.close()

elif values['well'] == True:
    window = sg.Window('Parameters', layout_well).Finalize()
    
    event, values = window.read()

    window.close()

    run_options = ['BOTH', 'SEGMENTATION', 'TRACKING']
    cond_run = [values['segm_track'], values['segm_only'], values['track_only']]
    ix_cond = np.where(np.array(cond_run))[0][0]
    
    file_settings = open("settings.sh","w") 
    file_settings.write("RUN_OPTION=" + run_options[ix_cond] + "\n") 
    file_settings.write("START_FRAME=" + values['start_frame'] + "\n") 
    file_settings.write("END_FRAME=" + values['end_frame'] + "\n") 
    file_settings.write("DATA_TYPE=WELL" + "\n") 
    file_settings.write("PATH_FILE=" + values['file_name'] + "\n") 
    file_settings.write("FILE_TYPE=" + values['file_type'] + "\n") 
    file_settings.write("MATLAB_ROOT=" + values['matlab_root'] + "\n") 
    file_settings.write("CONSTANTS=" + values['constants'] + "\n") 
    file_settings.write("TIME_STEP=" + values['time_step'] + "\n") 
    file_settings.write("NEIGHBOR_FLAG=" + values['neighbor_flag'] + "\n") 
    file_settings.write("MIN_CELL_AGE=" + values['min_cell_age'] + "\n") 
    file_settings.close()
