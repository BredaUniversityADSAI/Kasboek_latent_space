import os
from datetime import datetime

def manual_deletion():
    cwd = os.getcwd()
    os.chdir("..")
    cwd = os.getcwd()
    PRINTS_LOCATION = f'{cwd}/docs/'

    for file in os.listdir(PRINTS_LOCATION):
        if 'print_' in file:
            os.remove(f'{PRINTS_LOCATION}{file}')
    os.chdir('kui')

def automatic_deletion():
    cwd = os.getcwd()
    os.chdir("..")
    cwd = os.getcwd()
    PRINTS_LOCATION = f'{cwd}/docs/'
    
    date = int(str(datetime.now().date()).replace('-', ''))
    for file in os.listdir(PRINTS_LOCATION):
        if 'print_' in file:
            if int(file[6:14]) < date:
                os.remove(f'{PRINTS_LOCATION}{file}')
    os.chdir('kui')
