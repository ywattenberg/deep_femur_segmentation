"""
Author   : Serena Bonaretti 
Date     : Created on 24 January 2023.  Last update: 21 April 2023
License  : GPL GNU v3.0  
Email    : serena.bonaretti.research@gmail.com  

This module contains code to get the rod coordinates.

The function get_rod_coordinates_img calls get_rod_coordinates_slice from the map() from the package multiprocessing
"""


import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

import multiprocessing
import time



def get_rod_coordinates_slice (img_and_i, n_of_rods = 6):
    
    """
    Given an image and an index, it extracts a slice, where it find the circles and it returns their centers and radii

    Parameters
    ----------
    img : 3D SimpleITK Image
        Grey level image, containing the Bone Density Calibration Phantom with 6 rods
    n_of_rods : integer
        Number of expected rods (i.e. circles)
    
    Returns
    -------
    cx : list of integers
        Contains the x coordinates of the centers of the HA rods
    
    cy : list of integers
        Contains the y coordinates of the centers of the HA rods
    
    radii : list of integers
        Contains the lengths of the radii of the HA rods    
    """

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

    img = img_and_i[0]
    i   = img_and_i[1]
    slice_sitk = img[:,:,i]

    # convert slice from simpleitk to numpy
    one_slice_np = sitk.GetArrayFromImage(slice_sitk)

    # get the edges with canny
    edges = canny(one_slice_np, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    
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

    # security check: check that all x coordinates of the circles are somehow aligned. if they are not, remove that rod and send a message [to be revised and improved]
    # find outlier
    # get lower quantile
    # lower_quantile = np.quantile(cy,0.1)
    # # calculate average of remaining rods
    # cy_average = 0
    # for element in cy:
    #     if element > lower_quantile:
    #         cy_average 

    # quantile_tolerance = lower_quantile - sum(cy)/len(cy) - 30

    # if lower_quantile < quantile_tolerance

    # print (cy)

    # final_cx = []
    # final_cy = []
    # final_radii = []
    # outlier_found = False
    # for i in range (len(cx)):
    #     if cy[i] >= lower_quantile: 
    #         final_cx.append(cx[i])
    #         final_cy.append(cy[i])
    #         final_radii.append(radii[i])
    #         outlier_found = True
    #     else: 
    #         print ("Skipped one calibration rod for wrong detection")
    
    # if outlier_found == False:
    #     final_cx = cx
    #     final_cy = cy
    #     final_radii = radii



    return i, cx, cy, radii
    # return final_cx, final_cy, final_radii, edges





def get_rod_coordinates_img (img, n_of_processes):

    """
    It paralellize the call of get_rod_coordinates_slice for all slices of an image

    Parameters
    ----------
    img: SimpleITK
        3D image
    n_of_processes: int
        N. of cores

    Returns
    -------
    rods_geometry: list of lists
        Each sub-list contains slice id (int) and cx (list), cy (list), and radii (list) of all rods in the slice
        E.g. [0, [79, 247, 416, 585, 754, 922], [709, 727, 736, 735, 727, 709], [44, 44, 44, 43, 44, 44]]
    """
    
    start_time = time.time()
    pool = multiprocessing.Pool(processes=n_of_processes)
    rods_geometry = pool.map(get_rod_coordinates_slice, [[img, i] for i in range(0,img.GetSize()[2])])

    print ("-> Completed")
    print ("-> The total time was %.2f seconds (about %d min)" % ((time.time() - start_time), (time.time() - start_time)/60))

    # Transform rods_geometry from a list of typles to a list of lists for better future manipulation (e.g. add list with intensities to each sub-list)
    for i in range (len(rods_geometry)):
        rods_geometry[i] = list(rods_geometry[i])

    return rods_geometry
