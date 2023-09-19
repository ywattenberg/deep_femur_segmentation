"""
Author   : Serena Bonaretti 
Date     : Created on 25 January 2023.  Last update: 21 April 2023
License  : GPL GNU v3.0  
Email    : serena.bonaretti.research@gmail.com  

This module contains functions to pre- and/or post- process PCCT images of bones
"""


import SimpleITK as sitk
import time

import numpy as np

    
def morph_operations_2D(img, radius = 5, show_feedback=True):
    """
    Morphing operation on single slice to clean segmentation. Operations are: dilate + fill + erode

    Parameters
    ----------

    img: SimpleITK image
        Image to be processed
    radius: int
        Radius of the circular kernel used by the morphological operators. Default is 5
    show_feedback : bool
        If true, prints out feedback with slice numeber every 10 slices, and computational time
    
    Returns
    -------
    filtered: SimpleITK image
        Processed image

    """
    
    # start clock and provide feedback
    if show_feedback == True:
        start = time.time()

    # initialize the output image
    filtered = img

    # initialize filters
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius(radius)
    fill_filter = sitk.BinaryFillholeImageFilter()
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius(radius)
    
    # get number of slices
    if len(img.GetSize()) == 3:
        n_of_slices = img.GetSize()[2]
    elif len(img.GetSize()) == 2:
        n_of_slices = 1

    for i in range (n_of_slices):

        # provide feedback
        if i%10 == 0:
            print (str(i), end="... ")

        # distinguishing between 2D and 3D input image for the next command
        if len(img.GetSize()) == 3:
            mask_slice = img[:,:,i]
        elif len(img.GetSize()) == 2:
            mask_slice = img

        # dilate
        mask_slice = dilate_filter.Execute(mask_slice)

        # fill holes
        mask_slice = fill_filter.Execute(mask_slice)

        # erode back to original size 
        mask_slice = erode_filter.Execute(mask_slice)

        # slice inserted into volume
        if len(img.GetSize()) == 3:
            filtered[:,:,i] = mask_slice
        elif len(img.GetSize()) == 2:
            filtered = mask_slice       

    # print computational time
    if show_feedback == True:
        end = time.time()
        print ("Computational time: " + str(round(end - start,2)) + "s / " + str(round((end - start)/60,2)) + "min" )
    
    # casting to make sure
    filtered = sitk.Cast(filtered, sitk.sitkUInt8)
    
    return filtered



