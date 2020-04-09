# utils.py
# Contains basic functions that will be used throughout the project

import numpy as np
import os
from HyperParameters import *
from datetime import datetime
from PIL import Image

# Averages the RGB values in an entire image array
# Input: An image array
# Output: A grayscale image array
def grayscale(arr):
    arr = arr[:,:,:3]
    return (arr[:,:,0] + arr[:,:,1] + arr[:,:,2]) / 3

def resize(arr, width, height):
    img = Image.fromarray(arr)
    img = img.resize((width, height))
    return np.array(img)

def preprocess(arr, img_width, img_height, channels):
    if channels == 1:
        arr = grayscale(arr)
    arr = resize(arr, img_width, img_height)
    return arr.reshape(arr.shape[1], arr.shape[0]) # Pillow returns it backwards for some reason


def huber_loss(target, prediction):
    delta = params['HUBER_DELTA']
    loss = np.power(target - prediction, 2) if (target - prediction) <= delta else np.abs(target - prediction)
    return loss

def map_actions(action):
    return action * 3

# Makes the logfile and returns its path
def log_create():
    time = datetime.utcnow()
    time_str = time.strftime('%d%m%y_%H:%M:%S')
    logpath = get_logpath() + time_str + '.log'

    log = open(logpath, 'w')
    log.write('Log created: %s\n\n' %time)
    log.close()

    return logpath

# Initializes the logfile
def log_params(logpath, summary):
    log = open(logpath, 'a')
    hParams = open('../../HyperParameters.py', 'r')
    
    # Logs the model summary
    log.write('SUMMARY:\n')
    log.write(summary)
    
    log.write('\n\nHYPER PARAMETERS:\n')
    for line in hParams:
        # To skip the first two lines because they're pointless
        if not (line.startswith('#') or line.startswith('\n')):
            log.write(line)

    log.write('\n\nOUTPUTS:\n')
    hParams.close()
    log.close()


# Will log and print all things desired. 
# Recommended for things that are intended to be logged and printed.
def log_output(logpath, *args):
    log = open(logpath, 'a')
    
    for arg in args:
        print(arg)
        log.write(arg + '\n')
    
    log.close()

# Finds the logpath and returns it (Should only be called by log_create() as it only returns a directory)
def get_logpath():
    # Finds where the log should go
    pathlist = os.getcwd().split('/')
    basepath = '/'.join(pathlist[:-2]) 
    logpath = basepath + '/logs/' + '/'.join(pathlist[-2:]) + '/'

    # Creates all folders
    if not os.path.isdir(basepath + '/logs'):
        os.mkdir(basepath + '/logs')
    
    if not os.path.isdir(basepath + '/logs/' + pathlist[-2]):
        os.mkdir(basepath + '/logs/' + pathlist[-2])

    if not os.path.isdir(logpath):
        os.mkdir(logpath)
    
    return logpath
