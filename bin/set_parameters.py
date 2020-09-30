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
          [sg.Checkbox('Chamber', key='chamber', default=False), sg.Checkbox('Well', key='well', default=False)],
	        [sg.Text('')],
          [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]

layout_chamber = [[sg.Text('Foldername', key = 'title_folder_name', font='bold')],
                  [sg.Column([[sg.Input(key='folder_name'), sg.FolderBrowse()]], key='col_folder_name')],
                  [sg.Text('Filetype (e.g. tif, tiff, ...)', key = 'title_file_type', font='bold')],
                  [sg.Input(key='file_type')],
                  [sg.Text('')],
                #   [sg.Text('Input Files', key = 'input_files', font='bold')],
                  [sg.Text('Identifier of phase channel (e.g. Phase, PH, ...)', key = 'phase_check', font='bold')],
                  [sg.Input(key='ch1')],
                  [sg.Text('')],
                #   [sg.Text('Additional channel 1', key = 'add_channel_1', font='bold')],
                  [sg.Text('Identifier of additional channel type 1 (e.g. eGFP, GFP, YFP, ...)', key = 'channel_1', font='bold')],
                  [sg.Input(key='ch2')],
                #   [sg.Text('Cell type 1', key = 'cell_type_1')],      
                #   [sg.Column([[sg.Checkbox('13B01', key='cell_type_11'), sg.Checkbox('ZF270g', key='cell_type_12'), sg.Checkbox('FS144', key='cell_type_13')],
                #   [sg.Checkbox('A3M17', key='cell_type_14'), sg.Checkbox('1F187', key='cell_type_15')]], key = 'col_cell_type_1')], 
                  [sg.Text('')],
                #   [sg.Text('Additional channel 2', key = 'add_channel_2', font='bold')],
                  [sg.Text('Identifier of additional channel type 2 (e.g. mCheery, TXRED, ...)', key = 'channel_2', font='bold')],
                  [sg.Input(key='ch3')],
                #   [sg.Text('Cell type 2', key = 'cell_type_2')],      
                #   [sg.Column([[sg.Checkbox('13B01', key='cell_type_21'), sg.Checkbox('ZF270g', key='cell_type_22'), sg.Checkbox('FS144', key='cell_type_23')],
                #   [sg.Checkbox('A3M17', key='cell_type_24'), sg.Checkbox('1F187', key='cell_type_25')]], key = 'col_cell_type_2')], 
                  [sg.Text('')],
                #   [sg.Text('Matlab root folder', font='bold')],
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
                  [sg.Input('3', key='min_cell_age')],
	                [sg.Text('')],
                  [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]
                
layout_well = [[sg.Text('Filename', key = 'title_file_name', font='bold')],
               [sg.Column([[sg.Input(key='file_name'), sg.FileBrowse()]], key='col_file_name')],
               [sg.Text('')],
            #    [sg.Text('Matlab root folder', font='bold')],
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

if values['chamber'] == True:
    window = sg.Window('Parameters', layout_chamber).Finalize()

    event, values = window.read()

    window.close()

    channel_type_vals = [values['ch1'], values['ch2'], values['ch3']]
    sel_channel_types = [x for x in channel_type_vals if x]
    #sel_channel_types = channel_type_vals

    # cell_types = ['13B01', 'Zf270g', 'FS144', 'A3M17', '1F187', '13B01', 'Zf270g', 'FS144', 'A3M17', '1F187']
    # cell_type_vals = [values['cell_type_11'], values['cell_type_12'], values['cell_type_13'], values['cell_type_14'], values['cell_type_15'],\
    #                 values['cell_type_21'], values['cell_type_22'], values['cell_type_23'], values['cell_type_24'], values['cell_type_25']]
    # ix_cells = np.where(cell_type_vals)[0]
    # sel_cell_types = [cell_types[i] for i in ix_cells]

    # dict_file = [{'FOLDERNAME' : values['folder_name']},
    # {'CHANNELS' : sel_channel_types}, {'CELL_TYPES': sel_cell_types}]

    dict_file = [{'FOLDERNAME' : values['folder_name']},
    {'CHANNELS' : sel_channel_types}]

    file_settings = open("settings.sh","w") 
    file_settings.write("DATA_TYPE=CHAMBER" + "\n") 
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
    
    file_settings = open("settings.sh","w") 
    file_settings.write("DATA_TYPE=WELL" + "\n") 
    file_settings.write("PATH_FILE=" + values['file_name'] + "\n") 
    file_settings.write("MATLAB_ROOT=" + values['matlab_root'] + "\n") 
    file_settings.write("CONSTANTS=" + values['constants'] + "\n") 
    file_settings.write("TIME_STEP=" + values['time_step'] + "\n") 
    file_settings.write("NEIGHBOR_FLAG=" + values['neighbor_flag'] + "\n") 
    file_settings.write("MIN_CELL_AGE=" + values['min_cell_age'] + "\n") 
    file_settings.close()
