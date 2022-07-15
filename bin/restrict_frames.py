import PySimpleGUI as sg
import yaml
import numpy as np

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

file_settings = open("settings.sh","a") 
file_settings.write("START_FRAME=" + str(int(values['start'])) + "\n") 
file_settings.write("END_FRAME=" + str(int(values['end'])) + "\n") 
file_settings.close()