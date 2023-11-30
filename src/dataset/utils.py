import numpy as np

def get_inital_crop_size(size):
    """ 
        Calculate the initial crops size such that after rotation and zoom we get artifacts. 
        Assume that the maximum rotation angle is 90 degrees and the minimum zoom is 0.9.
    """
    inner_length = np.ceil(np.sqrt((size[0] ** 2 + size[1] ** 2)))
    inital_crop_size = int(np.ceil(np.sqrt((inner_length ** 2 + size[2] ** 2))))*1.1+1
    return int(np.ceil(inital_crop_size))