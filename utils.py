# utils.py
# Contains basic functions that will be used throughout the project

import numpy as np
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

def preprocess(arr, channels, img_width, img_height):
    if channels == 1:
        arr = grayscale(arr)
    arr = resize(arr, img_width, img_height)
    return arr.reshape(arr.shape[1], arr.shape[0]) # Pillow returns it backwards for some reason
