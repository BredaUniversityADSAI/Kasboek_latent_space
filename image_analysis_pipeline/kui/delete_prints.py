import os
from datetime import datetime

def manual_deletion():
    PRINTS_LOCATION = f'{os.getcwd()}/docs/'
    for file in os.listdir(PRINTS_LOCATION):
        if 'print_' in file:
            os.remove(f'{PRINTS_LOCATION}{file}')

def automatic_deletion():
    PRINTS_LOCATION = f'{os.getcwd()}/docs/'
    date = int(str(datetime.now().date()).replace('-', ''))
    time = int(str(datetime.now().time())[:5].replace(':', ''))
    if time == 2359:
        for file in os.listdir(PRINTS_LOCATION):
            if f'print_{date}' in file:
                os.remove(f'{PRINTS_LOCATION}{file}')
