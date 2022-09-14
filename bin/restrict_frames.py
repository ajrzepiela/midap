import PySimpleGUI as sg
import re

# set layout for GUI
sg.theme('LightGrey1')
appFont = ("Arial", 12)
sg.set_options(font=appFont)

def Button(*args, **kwargs):
    return sg.Col([[sg.Button(*args, **kwargs)]], pad=(0,0))

layout = [[sg.Text('Restrict number of frames', font='bold')],
          [sg.Text('Start')],
          [sg.Input('0', key='start')],
          [sg.Text('End')],
          [sg.Input('100', key='end')],
	      [sg.Text('')],
          [sg.Column([[sg.OK(), sg.Cancel()]], key='col_final')]]

window = sg.Window('Frame number', layout).Finalize()

event, values = window.read()
window.close()

# Throw an error if we pressed cancel or X
if event == 'Cancel' or event == None:
   exit(1)

# replace the current start and end frame with the selected one
with open("settings.sh","r+") as file_settings:
    # read file
    content = file_settings.read()

    # replace
    content = re.sub("START_FRAME\=.*", f"START_FRAME={values['start']}", content)
    content = re.sub("END_FRAME\=.*", f"END_FRAME={values['end']}", content)

    # truncate, set stream to start and write
    file_settings.truncate(0)
    file_settings.seek(0)
    file_settings.write(content)
