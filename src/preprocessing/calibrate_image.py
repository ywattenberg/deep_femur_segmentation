import matplotlib.pyplot as plt
import numpy as np

import SimpleITK as sitk

import multiprocessing
import time
from functools import partial


NOMINAL_VALUES_6_RODS = [797.0, 601.2, 399.6, 201.4, 100.8, 0]

def calibrate_image_slice(image, rod_coordinates, n_of_rods=6):
    gray_values = rod_coordinates[-1]

    if len(gray_values) == 4:
        nominal_values = NOMINAL_VALUES_6_RODS[1:-1]
    elif len(gray_values) == 6:
        nominal_values = NOMINAL_VALUES_6_RODS.copy()
    print(nominal_values)
    m,b = np.polyfit(gray_values, nominal_values, 1)
    return m*image + b




def calibrate_image_list(images, rod_coordinates, n_of_rods=6, threads=None):
    """
    Calibrates a list of images to using linear regression to the nominal values of the rods in the calibration phantom

    Parameters
    ----------
    images : list of numpy arrays (2D)
        Grey level images, containing the Bone Density Calibration Phantom with 6 rods
    rod_coordiantes : list of tuples 
        Contains the coordinates of the centers of the HA rods and their radii
    n_of_rods : integer
        Number of expected rods (i.e. circles)
    threads : integer
        Number of threads to use for multiprocessing
    """

    if threads is None:
        threads = multiprocessing.cpu_count()

    with multiprocessing.Pool(threads) as pool:
        map_func = partial(calibrate_image_slice, n_of_rods=n_of_rods)
        results = pool.starmap(map_func, zip(images, rod_coordinates))
    
    return results

def morph_mask_2D(image, radius=5):
    filtered = sitk.GetImageFromArray(image.astype(np.uint8)*255)

    # initialize filters
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(radius)
    fill_filter = sitk.BinaryFillholeImageFilter()
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(radius)
    mask_slice = filtered
    # dilate
    mask_slice = dilate_filter.Execute(mask_slice)

    # fill holes
    mask_slice = fill_filter.Execute(mask_slice)

    # erode back to original size 
    mask_slice = erode_filter.Execute(mask_slice)

    mask = sitk.ConnectedComponent(mask_slice)
    mask = sitk.RelabelComponent(mask, sortByObjectSize=True)

    posterior = mask == 1

    filtered = sitk.GetArrayFromImage(mask).astype(np.uint8)
    return filtered


    
    

