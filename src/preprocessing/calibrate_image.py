import matplotlib.pyplot as plt
import numpy as np

import SimpleITK as sitk

import multiprocessing
import time
from functools import partial


NOMINAL_VALUES_6_RODS = [797.0, 601.2, 399.6, 201.4, 100.8, 0]

def calibrate_image_slice(image, rod_coordinates):
    num_of_rods = len(rod_coordinates[-1])
    gray_values = rod_coordinates[-1]

    if len(gray_values) == 4:
        nominal_values = NOMINAL_VALUES_6_RODS[1:-1]
    elif len(gray_values) == 6:
        nominal_values = NOMINAL_VALUES_6_RODS.copy()
    print(nominal_values)
    m,b = np.polyfit(gray_values, nominal_values, 1)
    return m*image + b


def calibrate_image_list(images, rod_coordinates, threads=None):
    """
    Calibrates a list of images to using linear regression to the nominal values of the rods in the calibration phantom

    Parameters
    ----------
    images : list of numpy arrays (2D)
        Grey level images, containing the Bone Density Calibration Phantom with 6 rods
    rod_coordiantes : list of tuples 
        Contains the coordinates of the centers of the HA rods and their radii
    threads : integer
        Number of threads to use for multiprocessing
    """

    if threads is None:
        threads = multiprocessing.cpu_count()

    with multiprocessing.Pool(threads) as pool:
        results = pool.starmap(calibrate_image_slice, zip(images, rod_coordinates))
    
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


def cut_calibration_phantom(image, rod_coordinates, buffer=0.2):
    """
    Cut the image off above the calibration phantom to reduce the size of the image.

    Parameters
    ----------
    image : numpy array (2D)
        Grey level image, containing the Bone Density Calibration Phantom
    rod_coordiantes : list of tuples
        Contains the coordinates of the centers of the HA rods and their radii (can also contain the gray values of the rods)
        Important: `y` has to be third element of the tuple and `radius` the fourth
    buffer : float
        Percentage of the radius of the largest rod to be used as a buffer to cut the image
    
    Returns
    -------
    image : numpy array (2D)
        Grey level image, containing the Bone Density w\o the calibration phantom
    
    """
    all_cy = rod_coordinates[2]
    all_radii = rod_coordinates[3]

    cy = (all_cy[0] + all_cy[-1])/2
    radius = (all_radii[0] )#+ all_radii[-1])/2
    cut_off = cy - radius - buffer*radius

    image = image[:int(cut_off), :]
    return image


def calc_cutoff_coordinate(rod_coordinates, buffer=0.2):
    all_cy = rod_coordinates[2]
    all_radii = rod_coordinates[3]

    cy = (all_cy[0] + all_cy[-1])/2
    radius = (all_radii[0] )#+ all_radii[-1])/2
    cut_off = cy - radius - buffer*radius
    return cut_off


def cut_images(images, rod_coordinate_list, cut_above=0, buffer=0.2):
    """
    Cut the image off above the calibration phantom to reduce the size of the image. Further `cut_above` pixels are cut off from the top.

    Parameters
    ----------
    images : list of numpy arrays (2D)
        Grey level images, containing the Bone Density Calibration Phantom
    rod_coordiantes_list : list of tuples
        Contains the coordinates of the centers of the HA rods and their radii (can also contain the gray values of the rods)
        Important: `y` has to be third element of the tuple and `radius` the fourth
    cut_above : integer
        Number of pixels to cut off from the top of the image
    buffer : float
        Percentage of the radius of the largest rod to be used as a buffer to cut the image

    Returns
    -------
    image : numpy array (2D)
        Grey level image, containing the Bone Density w\o the calibration phantom
    """
    # all_cy = rod_coordinates[2]
    # all_radii = rod_coordinates[3]
    # Calculate the average y coordinate for the last and first rod
    avg_cutoff = np.mean([calc_cutoff_coordinate(rod_coordinates, buffer=buffer) for rod_coordinates in rod_coordinate_list])
    
    return images[:, :int(avg_cutoff), :]

