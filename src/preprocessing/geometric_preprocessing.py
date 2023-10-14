import itk
import SimpleITK as sitk
import numpy as np
import os
import logging
from util import read_dicom_series, image_to_array, array_to_image

def reverse_slice_order(image: sitk.Image):
    # Get the image as a numpy array
    image_array = image_to_array(image)
    # Reverse the order of the slices
    image_array = image_array[::-1]
    # Convert the numpy array back to an itk image
    new_direction = image.GetDirection()
    image = array_to_image(image_array, image.GetSpacing(), image.GetOrigin(), image.GetDirection())
    
    return image

def cutout_cast_HR_pQCT(image: sitk.Image, pixel_value_interval: tuple = (0, 1000)):
    """
    This function cuts out the cast that is used to hold the sample in place during scanning.
    The function assumes that the cast is stable in the z-direction, and that the sample is not. Further it assumes that the first/last slices are empty except for the cast.

    Parameters
    ----------
    image : sitk.Image
        The image to cut out the cast from.
    
    Returns
    -------
    sitk.Image
        The image with the cast cut out.
    """
    # For the time just hard code the region to cut out
    # We choose the biggest possible square region that does not contain the cast
    LOWER_X, UPPER_X = 0, 1000
    LOWER_Y, UPPER_Y = 0, 1000

    image = image[LOWER_X:UPPER_X, LOWER_Y:UPPER_Y, :]
    return image
    
    
    # TODO: Make this more general


    # Find the first slice that is not empty
    # TODO: Find good base level for threshold in empty images 
    size = image.GetSize()
    threshold = size[0] * size[1] * 0.1

    for i in range(100):
        slice = sitk.GetArrayFromImage(image[:,:,i])
        if np.sum(slice) > threshold:
            first_slice = i
            break
    
    # Find the last slice that is not empty
    for i in range(100):
        slice = sitk.GetArrayFromImage(image[:,:,-i])
        if np.sum(slice) > threshold:
            last_slice = i
            break
    


    first_slices = sitk.GetArrayFromImage(image[:,:,0:10])
    last_slices = sitk.GetArrayFromImage(image[:,:,-10:])
    first_slice = np.mean(first_slices, axis=2)
    last_slice = np.mean(last_slices, axis=2)

    def cutout_calibration_phantom(image : sitk.Image, calibration_rods: list = None):
        """
        This function cuts out the calibration phantom that is used to calibrate the scanner.
        We use the detection of the calibration phantom  rods to find the the position of the calibration phantom.
        """

        if calibration_rods is None:
            # Detect the calibration rods in every 50th slice of the image (for speed) as the coordinates of the rods are the relative stable in the z-direction
            calibration_rods = detect_calibration_rods(image[:,:,::50])
        

        

    
    
    


