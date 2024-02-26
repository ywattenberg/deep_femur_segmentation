import numpy as np
import os


def get_inital_crop_size(size):
    """ 
        Calculate the initial crops size such that after rotation and zoom we get artifacts. 
        Assume that the maximum rotation angle is 90 degrees and the minimum zoom is 0.9.
    """
    inner_length = np.ceil(np.sqrt((size[0] ** 2 + size[1] ** 2)))
    inital_crop_size = int(np.ceil(np.sqrt((inner_length ** 2 + size[2] ** 2))))*1.1+1
    return int(np.ceil(inital_crop_size))

def load_slices_region(path:str, roi:list):
    """ Load ROI gicne a path to a folder with slices (as numpy arrays) and a region of interest (roi)

    Args:
        path (str): Path to folder with slices
        roi (list of pairs): Region of interest (roi) in the format [(z_start, length), (x_start, length), (y_start, length)]

    Returns:
        np.array: ROI
    """        
    slices = range(roi[0][0], roi[0][0]+roi[0][1])
    x1, x_len = roi[1]
    y1, y_len = roi[2]

    images = []
    for i in slices:
        image = np.load(os.path.join(path, f"{i}.npy"))
        if i == slices[0]:
            print(f"Image shape: {image.shape}")
        images.append(image[x1:x1+x_len, y1:y1+y_len])
    images = np.stack(images, axis=0)
    return images


def load_slices(path:str, start:int, length:int):
    """ Load slices from a folder with slices (as numpy arrays)

    Args:
        path (str): Path to folder with slices
        start (int): Start slice
        length (int): Number of slices
    
    Returns:
        np.array: Slices
    """
    slices = range(start, start+length)
    images = []
    for i in slices:
        image = np.load(os.path.join(path, f"{i}.npy"))
        images.append(image)
    images = np.stack(images, axis=0)
    return images

def fix_mask_depth(mask, image_shape):
    """ Fix mask depth to match the image depth. """
    if image_shape[0] - mask.shape[0] > 10:
        raise ValueError("Mask depth difference is too big")
    if mask.shape[0] < image_shape[0]:
        padd = mask[0, :, :]
        padd = np.stack([padd]*(image_shape[0] - mask.shape[0]), axis=0)
        mask = np.concatenate([padd, mask], axis=0)
    return mask

def get_padded_mask(mask, image_shape, offset):
    padded_mask = np.zeros(image_shape, dtype=bool)
    padded_mask[0,:, offset[0]:offset[0]+mask.shape[1], offset[1]:offset[1]+mask.shape[2]] = mask
    return padded_mask

def maximum_overlapping_rectangle(rect1, rect2):
    """ Calculate the maximum overlapping rectangle between two rectangles. """
    (x,w), (y,h) = rect1
    (x2,w2), (y2,h2) = rect2
    x_overlap = max(0, min(x+w, x2+w2) - max(x, x2))
    y_overlap = max(0, min(y+h, y2+h2) - max(y, y2))
    intersection = [(max(x,x2), x_overlap),(max(y,y2), y_overlap)]
    return intersection



