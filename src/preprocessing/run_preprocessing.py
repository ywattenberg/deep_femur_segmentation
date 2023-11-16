# Imports
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import numpy as np
import sys
import logging

# Custom imports
from utils import read_dicom_series, image_to_array, array_to_image
from preprocessing import downsample_image, reverse_slice_order, cutout_cast_HR_pQCT, cutout_calibration_phantom, outlier_elimination, reorient_PCCT_image, intensity_normalization
from multi_part_image import split_series, merge_series, merge_sitk_images

def main(HR_pQCT_image_path, PCCT_image_path, cutoff_PCCT=1500):
    assert os.path.isdir(HR_pQCT_image_path), f"HR-pQCT image path {HR_pQCT_image_path} is not a directory."
    assert os.path.isdir(PCCT_image_path), f"PCCT image path {PCCT_image_path} is not a directory."
    path_to_subfolders = split_series(HR_pQCT_image_path, 3)
    logging.info(f"Splitting HR-pQCT image into {len(path_to_subfolders)} subfolders.")

    down_sampled_images = []
    for path in path_to_subfolders:
        image = read_dicom_series(path)
        # display the first and last slice of the image
        sitk.GetArrayFromImage

        logging.info(f" path: {path}, size: {image.GetSize()}, spacing: {image.GetSpacing()}")

        # Downsample the image
        image = downsample_image(image, 2)

        logging.info(f" path: {path}, size: {image.GetSize()}, spacing: {image.GetSpacing()}")

        down_sampled_images.append(image)

    image = merge_sitk_images(down_sampled_images)

    image = cutout_cast_HR_pQCT(image)

    image = intensity_normalization(image)

    image.SetDirection([1,0,0,0,1,0,0,0,1])
    image.SetOrigin([0,0,0])
    sitk.WriteImage(image, os.path.join(HR_pQCT_image_path, "downsampled_image_cut.nii.gz"))

    logging.info(f"Done with HRpQCT image")

    image = read_dicom_series(PCCT_image_path)
    image = image[:,:,:image.GetDepth()-300]
    image = cutout_calibration_phantom(image)
    image = reorient_PCCT_image(image)
    image = image[:,:,:cutoff_PCCT]
    image = intensity_normalization(image)
    image.SetDirection([1,0,0,0,1,0,0,0,1])
    image.SetOrigin([0,0,0])
    sitk.WriteImage(image, os.path.join(PCCT_image_path, "cutout_calibration_phantom_new.nii.gz"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    HR_pQCT_image_path = sys.argv[1]
    PCCT_image_path = sys.argv[2]
    if len(sys.argv) > 3:
        cutoff_PCCT = int(sys.argv[3])
    else:
        cutoff_PCCT = 1500
    main(HR_pQCT_image_path, PCCT_image_path, cutoff_PCCT=cutoff_PCCT)






