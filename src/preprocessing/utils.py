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
    # If the image is 3D, transpose the array to match the shape of the image
    if len(array.shape) == 3:
        array = array.transpose(1,2,0)
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
    if len(array.shape) == 3:
        image = sitk.GetImageFromArray(array.transpose(2,0,1))
    else:
        image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image

