import os
import SimpleITK as sitk
import numpy as np
from torch.nn.functional import interpolate
import argparse

def load_mask(mask_path, spacing, size):
    mask_slices = os.listdir(mask_path)
    mask_slices = sorted([slice for slice in mask_slices if slice.endswith(".npy")])
    mask = []
    for slice in mask_slices:
        mask.append(np.load(os.path.join(mask_path, slice)))
    mask = np.stack(mask)
    mask = interpolate(mask, size=size, mode="nearest")
    mask = sitk.GetImageFromArray(mask)
    mask.SetSpacing(spacing)
    mask.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    mask.SetOrigin([0, 0, 0])
    return mask

def convert_dataset_to_nnUNet_style(data_folder, out_folder):
    # Get subdirectories in data_folder
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "labelsTr"), exist_ok=True)

    # Image dir 
    image_dir = os.path.join(data_folder, "PCCT")
    # Mask dir
    mask_dir = os.path.join(data_folder, "HRpQCT")
    # Get subdirectories
    image_subdirs = os.listdir(image_dir)
    mask_subdirs = os.listdir(mask_dir)
    # Sort subdirectories
    image_subdirs = sorted([subdir for subdir in image_subdirs if os.path.isdir(os.path.join(image_dir, subdir))])
    mask_subdirs = sorted([subdir for subdir in mask_subdirs if os.path.isdir(os.path.join(mask_dir, subdir))])
    # Iterate over subdirectories
    for image_subdir, mask_subdir in zip(image_subdirs, mask_subdirs):
        # Get name of sample:
        name = os.path.basename(image_subdir)
        print(f"Processing {name}")
        image_path = os.path.join(image_dir, image_subdir, "result.mha")
        mask_path = os.path.join(mask_dir, mask_subdir, "mask")
        # Load images
        if os.path.exists(image_path) and os.pardir.exists(mask_path):
            print(f"Image and Mask exist for {name}")
            image = sitk.ReadImage(image_path)
            print(f"Image size: {image.GetSize()}")
            size_z = [image.GetSize()[2], image.GetSize()[0], image.GetSize()[1]]
            spacing = image.GetSpacing()
            mask = load_mask(mask_path, spacing, size_z)
            print(f"Mask size: {mask.GetSize()}")
            # Save images
            sitk.WriteImage(image, os.path.join(out_folder, "imagesTr", f"{name}.nii.gz"))
            sitk.WriteImage(mask, os.path.join(out_folder, "labelsTr", f"{name}.nii.gz"))
        else:
            print(f"Image or Mask does not exist for {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    args = parser.parse_args()
    convert_dataset_to_nnUNet_style(args.data_folder, args.out_folder)


        




            