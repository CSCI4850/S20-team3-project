# utils.py
# Contains basic functions that will be used throughout the project


# Averages the RGB values in an entire image array
# Input: An image array
# Output: A grayscale image array
def grayscale(arr):
	arr = arr[:,:,:3]
    return (arr[:,:,0] + arr[:,:,1] + arr[:,:,2]) / 3
