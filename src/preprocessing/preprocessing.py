import SimpleITK as sitk
import numpy as np
import os
import logging
import skimage

from utils import read_dicom_series, image_to_array, array_to_image
from calibration_phantom import get_rod_coordinates_mp

def downsample_image(image: sitk.Image, factor:int):
    """
    This function downsamples an image by a `factor` using the resampling filter. Notice that we downsample in all directions (x,y,z) by the same factor. Thus a factor of 2 will result in an image that is 8 (2^3) times smaller. This is to preserve the aspect ratio of the image.
    Further the function assumes that default pixel value is 0.

    Parameters
    ----------
    image : sitk.Image
        The image to downsample.
    factor : int
        The factor to downsample the image by.
    
    Returns
    -------
    itk.Image
        The downsampled image.

    """
    image_space = image.GetSpacing()
    image_size = image.GetSize()
    image_origin = image.GetOrigin()

    new_spacing = [space*factor for space in image_space]
    new_size = [int(size/factor) for size in image_size]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(image_origin)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetDefaultPixelValue(0)

    return resample.Execute(image)

def reverse_slice_order(image: sitk.Image):
    """
    This function reverses the order of the slices in an image. This is useful when the image is read in the wrong order.

    Parameters
    ----------
    image : sitk.Image
        The image to reverse the slice order of.

    Returns
    -------
    sitk.Image
        The image with the slice order reversed.
    """
    # Get the image as a numpy array
    image_array = image_to_array(image)
    # Reverse the order of the slices
    image_array = image_array[:,:,::-1]
    # Convert the numpy array back to an itk image
    new_direction = image.GetDirection()
    image = array_to_image(image_array, image.GetSpacing(), image.GetOrigin(), image.GetDirection())
    
    return image

def cutout_cast_HR_pQCT(image: sitk.Image, fill_value: int = None):
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
    array = image_to_array(image)
    first_slice = array[:,:,0]
    max_value = np.max(first_slice)
    min_value = np.min(first_slice)

    
    if min_value != 0:
        logging.warning("The pixel values of the image are not normalized to a range of [0,x]. This might cause problems when cutting out the cast.")
        normalized = False
        threshold = max_value * 0.1
    else:
        # If the image is normalized to a range of [0,x] we can use a higher threshold
        threshold = max_value * 0.2
        normalized = True


    mask = first_slice > threshold
    
    # Use morphological closing to fill in holes in the mask and dilation to make the mask larger
    mask = skimage.morphology.closing(mask, skimage.morphology.disk(5))
    mask = skimage.morphology.dilation(mask, skimage.morphology.disk(5))    

    inverse_mask = np.logical_not(mask)

    if not normalized and fill_value is None:
        logging.warning("The image is not normalized and no fill value is given. Using the minimum of voxel value as fill value.")
         
        # average the values of the first slice without the cast
        fill_value = np.mean(first_slice[inverse_mask])
    
    # Set the values of the cast to the fill value for every slice
    
    mask = np.tile(mask, (array.shape[2], 1, 1)).transpose(1,2,0)
    array[mask == 1] = fill_value

    # Convert the array back to an image
    image = array_to_image(array, image.GetSpacing(), image.GetOrigin(), image.GetDirection())

    return image


def cutout_calibration_phantom(image : sitk.Image, calibration_rods: list = None):
    """
    This function cuts out the calibration phantom that is used to calibrate the scanner.
    We use the detection of the calibration phantom rods to find the the position of the calibration phantom.

    Parameters
    ----------
    image : sitk.Image
        The image to cut out the calibration phantom from.
    calibration_rods : list, optional
        The coordinates of the calibration rods, by default None

    Returns
    -------
    sitk.Image
        The image with the calibration phantom cut out.
    """

    if calibration_rods is None:
        # Detect the calibration rods in every 50th slice of the image (for speed) as the coordinates of the rods are the relative stable in the z-direction
        calibration_rods = get_rod_coordinates_mp(image[:,:,::50])

    # We assume that the calibration phantom array is of the form [[x,y,radius], [x,y,radius], ...]
    # As a cutoff point we use the mean of the first and last rod y coordinate averaged in z direction
    # plus a small offset
    mean_y = 0
    for rod in calibration_rods:
        mean_y += rod[2][0] + rod[2][-1]
    mean_y = (mean_y*0.5) / len(calibration_rods)
    cutoff = int(mean_y* 0.87)
    image = image[:,:cutoff,:]
    return image

def outlier_elimination(image: sitk.Image, new_max_value: int = 100_000_000, fill_value: int = None):
    """
    This function eliminates outlier voxels from an image. 

    Parameters
    ----------
    image : sitk.Image
        The image to eliminate outliers from.
    mass_threshold : float, optional
        The threshold for eliminating outliers, by default 0.1
    
    Returns
    -------
    sitk.Image
        The image with the outliers eliminated.
    """
    

    new_max_value = 10_000
    ratio         = 0.0
    num_outliers  = 0
    num_iter      = 0
    max_num_iter  = 100
    thresh        = 0.00001

    # get image characteristics for numpy-SimpleITK conversion
    spacing   = image.GetSpacing()
    origin    = image.GetOrigin ()
    direction = image.GetDirection()

    # convert image from SimpleITK to numpy
    img_py = sitk.GetArrayFromImage(image)

    # get image max and min values
    max_value = np.max(img_py)
    min_value = np.min(img_py)

    if min_value < 0:
        img_py = img_py - min_value
        max_value = np.max(img_py)
        min_value = np.min(img_py)

    # set other variables
    cut_value = max_value
    num_total = image.GetNumberOfPixels()

    # calculate cut_value
    while ((ratio < thresh) and (num_iter < max_num_iter)):
        print(num_iter)
        print(ratio)
        num_iter     = num_iter + 1
        num_outliers = 0

        # calculate cut_value
        cut_value    = (cut_value + min_value) * 0.95

        # get the number of voxels that are above the cut_value (outliers)
        outliers    = img_py[img_py > cut_value]
        num_outliers = num_outliers + len(outliers)

        # calculate the new proportion of outlier voxels over the total number of voxles
        ratio = num_outliers / num_total

    # assign new voxel values
    img_py[img_py < cut_value] = img_py[img_py < cut_value] / cut_value * new_max_value
    img_py[img_py > cut_value] = new_max_value

    # back to SimpleITK
    image = sitk.GetImageFromArray (img_py)
    image.SetSpacing  (spacing)
    image.SetOrigin   (origin)
    image.SetDirection(direction)

    return image
    # array[array > cut_off] = new_max_value

    # Convert the array back to an image
    image = array_to_image(array, image.GetSpacing(), image.GetOrigin(), image.GetDirection())

    return image


def intensity_normalization(image: sitk.Image, lower_bound: int = 0, upper_bound: int = 2000):
    """
    This function normalizes the intensity of an image to a given range. This is useful when the intensity of the image is not normalized.

    Parameters
    ----------
    image : sitk.Image
        The image to normalize.
    lower_bound : int, optional
        The lower bound of the normalization, by default 0
    upper_bound : int, optional
        The upper bound of the normalization, by default 1000

    Returns
    -------
    sitk.Image
        The normalized image.
    """
    # Get the image as a numpy array
    array = image_to_array(image)
    # Normalize the array
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    # Scale the array to the given range
    array = array * (upper_bound - lower_bound) + lower_bound
    # Convert the array back to an image
    image = array_to_image(array, image.GetSpacing(), image.GetOrigin(), image.GetDirection())
    
    return image


def reorient_PCCT_image(image : sitk.Image):
    array = image_to_array(image)
    array = array[:,::-1,::-1]
    array = np.rot90(array, k=-1)
    image = array_to_image(array, image.GetSpacing(), image.GetOrigin(), image.GetDirection())
    return image


        

    
    
    


