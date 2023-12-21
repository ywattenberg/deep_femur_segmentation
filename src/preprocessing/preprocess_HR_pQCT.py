# Imports
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import numpy as np

# Custom imports
from .utils import read_dicom_series, image_to_array, array_to_image
from .preprocessing import downsample_image, reverse_slice_order, cutout_cast_HR_pQCT, cutout_calibration_phantom, outlier_elimination, reorient_PCCT_image, intensity_normalization, threshold_image
from .multi_part_image import split_series, merge_series, merge_sitk_images


FOLDER = 'data/HR-pQCT/'

def main():
    subfolders = os.listdir(FOLDER)
    subfolders.sort()
    subfolders = [os.path.join(FOLDER, subfolder) for subfolder in subfolders]
    subfolders = [subfolder for subfolder in subfolders if os.path.isdir(subfolder) and (subfolder.endswith("L") or subfolder.endswith("R"))]
    subfolders = [r"C:\Users\Yannick\Documents\repos\deep_femur_segmentation\data\HR-pQCT\2209_05992_L"]
    print(subfolders)
    for subfolder in subfolders:
        dirs = os.listdir(subfolder)
        if "part_0" in dirs:
            print(f"Skipping {subfolder}")
            continue

        path_to_subfolders = split_series(subfolder, 3)
        print(path_to_subfolders)

        path_to_subfolders = os.listdir(subfolder)
        path_to_subfolders = [os.path.join(subfolder, path) for path in path_to_subfolders if path.startswith("part")]
        print(path_to_subfolders)
        down_sampled_images = []
        for path in path_to_subfolders:
            image = read_dicom_series(path)
            # display the first and last slice of the image
            fig, ax = plt.subplots(1,2)
            sitk.GetArrayFromImage
            ax[0].imshow(sitk.GetArrayFromImage(image[:,:,0]))
            ax[1].imshow(sitk.GetArrayFromImage(image[:,:,-1]))
            plt.savefig(os.path.join(subfolder, "first_last_slice.png"))

            print(f" path: {path}, size: {image.GetSize()}, spacing: {image.GetSpacing()}")

            # Downsample the image
            image = downsample_image(image, 2)

            print(f" path: {path}, size: {image.GetSize()}, spacing: {image.GetSpacing()}")

            down_sampled_images.append(image)

            image = merge_sitk_images(down_sampled_images)

            no_cast, mask = cutout_cast_HR_pQCT(image[:,:,::50])

            plt.imshow(mask[:,:,0], cmap="gray")
            plt.savefig(os.path.join(subfolder, "mask.png"))

            #  Show the first slice of the image
            for i in range(0, no_cast.GetDepth(), 5):
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(sitk.GetArrayFromImage(image[:,:,i*50]), cmap="gray")
                ax[1].imshow(sitk.GetArrayFromImage(no_cast[:,:,i]), cmap="gray")
                plt.savefig(os.path.join(subfolder, f"compare_cast_removal_{i}.png"))

            image, _ = cutout_cast_HR_pQCT(image)

            image.SetDirection([1,0,0,0,1,0,0,0,1])
            image.SetOrigin([0,0,0])

            sitk.WriteImage(image, os.path.join(subfolder, "downsampled_image_cut_cast.nii.gz"))


if __name__ == "__main__":
    main()