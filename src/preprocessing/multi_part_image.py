import SimpleITK as sitk
import numpy as np
import os
import logging
import shutil

from .utils import read_dicom_series, image_to_array, array_to_image


def split_series(path: str, num_of_parts: int):
    """
    Split a DICOM series into multiple parts. This is useful when the DICOM series is too large to be read into memory. Resulting subdirectories will be named `_part_0`, `_part_1`, etc. Important: The DICOM files will be moved not copied to save space.

    Parameters
    ----------
    path : str
        Path to the directory containing the DICOM series.
    num_of_parts : int
        Number of parts to split the DICOM series into.

    Returns
    -------
    list
        List of paths to the subdirectories containing the parts of the DICOM series.
    """
    assert os.path.isdir(path), f"Path {path} is not a directory."
    assert num_of_parts > 0, f"Number of parts must be greater than 0."

    # Create subdirectories
    for i in range(num_of_parts):
        os.makedirs(os.path.join(path, f"part_{i}"), exist_ok=True)
    
    files = os.listdir(path)
    print
    # File is a DICOM file if it ends with .dcm or .DCM_1 (for legacy reasons)
    image_files = [file for file in files if file.lower().endswith(".dcm") or file.lower().endswith(".dcm_1")]
    image_files.sort()
    num_of_images = len(image_files)
    num_of_images_per_part = num_of_images // num_of_parts
    remainder = num_of_images % num_of_parts

    # Move files to subdirectories
    for i in range(num_of_parts):
        # Get the files for this part
        if i < remainder:
            start = i * (num_of_images_per_part + 1)
            end = start + num_of_images_per_part + 1
        else:
            start = i * num_of_images_per_part + remainder
            end = start + num_of_images_per_part
        files = image_files[start:end]
        # Move the files
        for file in files:
            os.rename(os.path.join(path, file), os.path.join(path, f"part_{i}", file))

    # Lastly copy the isq file to all subdirectories (if it exists)
    isq_file = [file for file in files if file.endswith(".isq") or file.endswith(".ISQ")]
    if len(isq_file) > 0:
        isq_file = isq_file[0]
        for i in range(num_of_parts):
            shutil.copy(os.path.join(path, isq_file), os.path.join(path, f"part_{i}", isq_file))
    
    return [os.path.join(path, f"part_{i}") for i in range(num_of_parts)]

def merge_series(base_path: str):
    """
    This function merges a DICOM series that was split using the `split_series` function. The resulting DICOM series will be saved in the base directory. The subdirectories will be deleted.

    Parameters
    ----------
    base_path : str
        Path to the directory containing the DICOM series.
    """

    # Find the subdirectories
    files = os.listdir(base_path)
    subdirectories = [file for file in files if os.path.isdir(os.path.join(base_path, file) and file.startswith("part_"))]
    num_of_parts = len(subdirectories)

    # Move the files to the base directory
    for subdirectory in subdirectories:
        files = os.listdir(os.path.join(base_path, subdirectory))
        for file in files:
            # Delete the ISQ file
            if file.endswith(".isq") or file.endswith(".ISQ"):
                os.remove(os.path.join(base_path, file))
            else:
                os.rename(os.path.join(base_path, subdirectory, file), os.path.join(base_path, file))
        os.rmdir(os.path.join(base_path, subdirectory))
    
def merge_sitk_images(images: list[sitk.Image]):
    """
    Merge a list of SimpleITK images into a single SimpleITK image.

    Parameters
    ----------
    images : list(sitk.Image)
        The images to merge.

    Returns
    -------
    sitk.Image
        The merged image.
    """
    # Convert the images to arrays
    arrays = [image_to_array(image) for image in images]

    # Merge the arrays
    array = np.concatenate(arrays, axis=2)

    # Convert the array to an image
    image = array_to_image(array, images[0].GetSpacing(), images[0].GetOrigin(), images[0].GetDirection())

    return image

