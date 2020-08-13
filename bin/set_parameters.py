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
          [sg.OK(), sg.Cancel()],
          #[sg.Text('')],
          [sg.Text('Foldername', key = 'title_folder_name', font='bold')],
          [sg.Column([[sg.Input(key='folder_name'), sg.FolderBrowse()]], key='col_folder_name')],
          [sg.Text('Filename', key = 'title_file_name', font='bold')],
          [sg.Column([[sg.Input(key='file_name'), sg.FileBrowse()]], key='col_file_name')],
          [sg.Text('Input Files', key = 'input_files', font='bold')],
          [sg.Text('Phase', key = 'phase_check')],
          #[sg.Input(key='inp_1'), sg.FileBrowse()],
          [sg.Column([[sg.Checkbox('Phase', key='phase', default=False), sg.Checkbox('PH', key='ph', default=False)]], key='col_phase')],
          [sg.Text('')],
          [sg.Text('Additional channel 1', key = 'add_channel_1', font='bold')],
          #[sg.Input(key='inp_2'), sg.FileBrowse()],
          [sg.Text('Channel type 1', key = 'channel_1')],
          [sg.Column([[sg.Checkbox('GFP', key='gfp_1'), sg.Checkbox('mCherry', key='cherry_1'), sg.Checkbox('TXRED', key='txred_1')]], key = 'col_add_ch_1')],
          [sg.Text('Cell type 1', key = 'cell_type_1')],      
          [sg.Column([[sg.Checkbox('13B01', key='cell_type_11'), sg.Checkbox('ZF270g', key='cell_type_12'), sg.Checkbox('FS144', key='cell_type_13')],
          [sg.Checkbox('A3M17', key='cell_type_14'), sg.Checkbox('1F187', key='cell_type_15')]], key = 'col_cell_type_1')], 
          [sg.Text('')],
          [sg.Text('Additional channel 2', key = 'add_channel_2', font='bold')],
          #[sg.Input(key='inp_3'), sg.FileBrowse()],
          [sg.Text('Channel type 2', key = 'channel_2')],
          [sg.Column([[sg.Checkbox('GFP', key='gfp_2'), sg.Checkbox('mCherry', key='cherry_2'), sg.Checkbox('TXRED', key='txred_2')]], key = 'col_add_ch_2')],
          [sg.Text('Cell type 2', key = 'cell_type_2')],      
          [sg.Column([[sg.Checkbox('13B01', key='cell_type_21'), sg.Checkbox('ZF270g', key='cell_type_22'), sg.Checkbox('FS144', key='cell_type_23')],
          [sg.Checkbox('A3M17', key='cell_type_24'), sg.Checkbox('1F187', key='cell_type_25')]], key = 'col_cell_type_2')], 
	      [sg.Text('')],
          [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]

window = sg.Window('Parameters', layout).Finalize()
#window = sg.Window('Parameters', resizable=True).Layout(layout).Finalize()
window.Element('title_folder_name').Update(visible=False)
window.Element('col_folder_name').Update(visible=False)
window.Element('title_file_name').Update(visible=False)
window.Element('col_file_name').Update(visible=False)
window.Element('input_files').Update(visible=False)
window.Element('phase_check').Update(visible=False)
window.Element('col_phase').Update(visible=False)
window.Element('add_channel_1').Update(visible=False)
window.Element('channel_1').Update(visible=False)
window.Element('col_add_ch_1').Update(visible=False)
window.Element('cell_type_1').Update(visible=False)
window.Element('col_cell_type_1').Update(visible=False)
window.Element('add_channel_2').Update(visible=False)
window.Element('channel_2').Update(visible=False)
window.Element('col_add_ch_2').Update(visible=False)
window.Element('cell_type_2').Update(visible=False)
window.Element('col_cell_type_2').Update(visible=False)
window.Element('col_final').Update(visible=False)
window.Refresh()
window.Refresh()

event, values = window.read()

if values['chamber'] == True:
    window.Element('title_folder_name').Update(visible=True)
    window.Element('col_folder_name').Update(visible=True)
    window.Element('phase_check').Update(visible=True)
    window.Element('col_phase').Update(visible=True)
    window.Element('add_channel_1').Update(visible=True)
    window.Element('channel_1').Update(visible=True)
    window.Element('col_add_ch_1').Update(visible=True)
    window.Element('cell_type_1').Update(visible=True)
    window.Element('col_cell_type_1').Update(visible=True)
    window.Element('add_channel_2').Update(visible=True)
    window.Element('channel_2').Update(visible=True)
    window.Element('col_add_ch_2').Update(visible=True)
    window.Element('cell_type_2').Update(visible=True)
    window.Element('col_cell_type_2').Update(visible=True)
    window.Element('col_final').Update(visible=True)
    window.Refresh()
    window.Refresh()

    event, values = window.read()

    window.close()

    # extract results from GUI
    # sel_inputs = []
    # for i in range(3):
    #     if len(values['inp_' + str(i + 1)]) > 0:
    #         sel_inputs.append(values['inp_' + str(i + 1)])

    channel_types = ['Phase', 'PH', 'GFP', 'mCherry', 'TXRED', 'GFP', 'mCherry', 'TXRED']
    channel_type_vals = [values['phase'], values['ph'], values['gfp_1'], values['cherry_1'], values['txred_1'], values['gfp_2'], values['cherry_2'], values['txred_2']]
    ix_channels = np.where(channel_type_vals)[0]
    sel_channel_types = [channel_types[i] for i in ix_channels]

    cell_types = ['13B01', 'Zf270g', 'FS144', 'A3M17', '1F187', '13B01', 'Zf270g', 'FS144', 'A3M17', '1F187']
    cell_type_vals = [values['cell_type_11'], values['cell_type_12'], values['cell_type_13'], values['cell_type_14'], values['cell_type_15'],\
                    values['cell_type_21'], values['cell_type_22'], values['cell_type_23'], values['cell_type_24'], values['cell_type_25']]
    ix_cells = np.where(cell_type_vals)[0]
    sel_cell_types = [cell_types[i] for i in ix_cells]

    dict_file = [{'FOLDERNAME' : values['folder_name']},
    {'CHANNELS' : sel_channel_types}, {'CELL_TYPES': sel_cell_types}]

    # save parameters to config file
    # with open(r'config.yaml', 'w') as file:
    #     documents = yaml.dump(dict_file, file)

    file_settings = open("settings.sh","w") 
    file_settings.write("DATA_TYPE=CHAMBER" + "/ \n") 
    file_settings.write("PATH_FOLDER=" + values['folder_name'] + "/ \n") 

    #file_settings.write("INPUTS=" + str(sel_inputs) + "\n")

    # for i, s in enumerate(sel_inputs):
    #     file_settings.write("INP_" + str(i + 1) + "=" + os.path.basename(s) + "\n")
    # file_settings.write("NUM_INP=" + str(len(sel_inputs)) + "\n")

    for i, s in enumerate(sel_cell_types):
        file_settings.write("CELL_TYPE_" + str(i + 1) + "=" + s + "\n")
    file_settings.write("NUM_CELL_TYPES=" + str(len(sel_cell_types)) + "\n")

    for i, s in enumerate(sel_channel_types):
            file_settings.write("CHANNEL_" + str(i + 1) + "=" + s + "\n")
    file_settings.write("NUM_CHANNEL_TYPES=" + str(len(sel_channel_types)) + "\n")
    file_settings.close()

if values['well'] == True:
    window.Element('title_file_name').Update(visible=True)
    window.Element('col_file_name').Update(visible=True)
    window.Element('col_final').Update(visible=True)
    window.Refresh()
    window.Refresh()
    
    event, values = window.read()

    window.close()
    
    file_settings = open("settings.sh","w") 
    file_settings.write("DATA_TYPE=WELL" + "\n") 
    file_settings.write("PATH_FILE=" + values['file_name'] + "\n") 
    file_settings.close()
