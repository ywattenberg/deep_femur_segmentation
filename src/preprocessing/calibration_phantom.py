import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

import multiprocessing
import time
from functools import partial

def _get_rod_coordinates(image, index, n_of_rods=4):
    """
    Given a image series it returns the centers and radii of the rods

    Parameters
    ----------
    img : numpy array (2D)
        Grey level image, containing the Bone Density Calibration Phantom with 6 rods
    index : integer
        Slice id of the image 
    n_of_rods : integer
        Number of expected rods (i.e. circles)
    
    Returns
    -------
    index : integer
        Slice id of the image

    cx : list of integers
        Contains the x coordinates of the centers of the HA rods
    
    cy : list of integers
        Contains the y coordinates of the centers of the HA rods
    
    radii : list of integers
        Contains the lengths of the radii of the HA rods    

    """
    # This function is a lifted from code by Serena Bonaretti (github.com/sbonaretti)
    # parameters
    sigma = 3
    low_threshold  = 1
    high_threshold = 10
    start_4 = 60 # rod radius is about 70
    stop_4  = 80 
    step_4  = 1
    start_6 = 30 # rod radius is about 40
    stop_6  = 50
    step_6  = 1
    min_circle_distance = 50

    img = sitk.GetArrayFromImage(image[:,:,index])

    edges = canny(img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    
    # create a range of cicle radiii for the transform
    if n_of_rods == 4:
        hough_radii = np.arange(start_4, stop_4, step_4)
    elif n_of_rods == 6:
        hough_radii = np.arange(start_6, stop_6, step_6)

    # extract ALL the circles from the image
    hough_res = hough_circle(edges, hough_radii)

    # select the 4 (4 rods phantom) or 5 (6rods phantom) most prominent cicles that have a distance of at least min_xdistance to avoid semi-duplicates
    # for 6 rod phantom, canny does not detect the 6th cicle (calculated manually below)
    
    if n_of_rods == 6:
        num_peaks = 5
    elif n_of_rods == 4:
        num_peaks = 4
    _, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=num_peaks, min_xdistance = min_circle_distance)

    # sort the circles according to x
    # get the indeces
    sorted_indeces = np.argsort(cx)
    # sort the lists
    cx = [cx[i] for i in sorted_indeces]
    cy = [cy[i] for i in sorted_indeces]
    radii = [radii[i] for i in sorted_indeces]
    
    # if there are 6 rods, the last rod is not visible (similar grey level as the phantom plastic), so it has to be defined manually
    if n_of_rods == 6:
        # x is the x of the last circle + mean of center distances
        center_distance = np.mean(np.array(cx[1:]) - np.array(cx[0:-1]))
        cx.append(cx[-1] + int(center_distance))
        # same y and radius 
        cy.append(cy[0])
        radii.append(radii[0])

    return index, cx, cy, radii

def get_rod_coordinates_mp(images, n_of_rods=4, threads=None):
    """
    Given a list of images it returns the centers and radii of the rods at every `stepsize` images using multiprocessing

    Parameters
    ----------
    images : list of numpy arrays (2D)
        Grey level images, containing the Bone Density Calibration Phantom with 6 rods
    n_of_rods : integer 
        Number of expected rods (i.e. circles)
    threads : integer
        Number of threads to use for multiprocessing. If None, the number of threads is set to the number of cores of the CPU

    Returns
    -------
    results: list of lists
        Each sub-list contains slice id (int) and cx (list), cy (list), and radii (list) of all rods in the slice
        E.g. [[79, 247, 416, 585, 754, 922], [709, 727, 736, 735, 727, 709], [44, 44, 44, 43, 44, 44]]
    """

    if threads is None:
        threads = multiprocessing.cpu_count()

    with multiprocessing.Pool(threads) as p:
        results = p.starmap(_get_rod_coordinates, zip([images] *images.GetDepth(), range(images.GetDepth())))

    return results


