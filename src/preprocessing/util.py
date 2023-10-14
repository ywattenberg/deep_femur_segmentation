import numpy as np
import SimpleITK as sitk
import os
import logging

def read_dicom_series(path: str) -> sitk.Image:
    """
    Read a DICOM series from a directory path.

    Parameters
    ----------
    path : str
        Path to the directory containing the DICOM series.
    
    Returns
    -------
    itk.Image
        The image read from the DICOM series.
    """
    # Assert that the path exists
    assert os.path.exists(path), f"Path {path} does not exist."

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    return image

def image_to_array(image: sitk.Image):
    """
    Convert an Simple-ITK image to a NumPy array.

    Parameters
    ----------
    image : sitk.Image
        The image to convert.
    
    Returns
    -------
    np.ndarray
        The converted image.
    """
    # Convert the image to a NumPy array
    array = sitk.GetArrayFromImage(image)
    array = np.transpose(array, (2, 1, 0))
    return array

def array_to_image(array: np.ndarray, spacing: tuple =[1,1,1], origin: tuple =[0,0,0], direction: tuple = [1,0,0,0,1,0,0,0,1]):
    """
    Convert a NumPy array to an SimpleITK image.

    Parameters
    ----------
    array : np.ndarray
        The array to convert.
    spacing : tuple, optional
        The spacing of the image, by default [1,1,1]
    origin : tuple
        The origin of the image, by default [0,0,0]
    direction : tuple
        The direction of the image, by default [1,0,0,0,1,0,0,0,1]
    
    Returns
    -------
    itk.Image
        The converted image.
    """
    if spacing == [1,1,1]:
        logging.WARNING("Spacing is not set, defaulting to [1,1,1]. This may cause issues.")

    # Convert the array to an ITK image and set the metadata
    image = sitk.GetImageFromArray(np.transpose(array, (2, 1, 0)))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image

def downsample_image(image: sitk.Image, factor:int):
    """
    This function downsamples an image by a `factor` using the resampling filter. Notice that we downsample in all directions (x,y,z) by the same factor. Thus a factor of 2 will result in an image that is 8 (2^3) times smaller. This is to preserve the aspect ratio of the image.
    Further the function assumes that default pixel value is 0.

    Parameters
    ----------
    image : itk.Image
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

