import os
import SimpleITK as sitk
import numpy as np
from torch.nn.functional import interpolate
import torch
import argparse

def load_mask(mask_path, spacing, size):
    mask_slices = os.listdir(mask_path)
    mask_slices = sorted([slice for slice in mask_slices if slice.endswith(".npy")], key=lambda x: int(x.split(".")[0]))
    mask = []
    for slice in mask_slices:
        mask.append(np.load(os.path.join(mask_path, slice)))
    mask = np.stack(mask)
    mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    mask = interpolate(mask, size=size, mode="nearest").squeeze()
    mask = mask.numpy()
    mask = sitk.GetImageFromArray(mask)
    mask.SetSpacing(spacing)
    mask.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    mask.SetOrigin([0, 0, 0])
    return mask

def convert_dataset_to_nnUNet_style(data_folder, out_folder):
    dict = { "1":[395, 835, 320, 1000],"3": [360,800,250,1010], "5":[315,850,375,955], "7":[330,775,285,970], "8":[330, 855, 265,1080], "12":[360,825,230,1000], "4":[299,858,392,1102], "6":[311,778,384,1110], "10":[377,795,411,1097], "11":[246,824,395,1087],"13":[370,800,309,954]}

    # Get subdirectories in data_folder
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "labelsTr"), exist_ok=True)

    # Image dir 
    image_dir = os.path.join(data_folder, "PCCT")
    # Mask dir
    mask_dir = r"D:\raw_data\HR-pQCT"
    # Get subdirectories
    image_subdirs = os.listdir(image_dir)

    # Sort subdirectories
    image_subdirs = sorted([subdir for subdir in image_subdirs if os.path.isdir(os.path.join(image_dir, subdir))])
    # Iterate over subdirectories
    for image_subdir in  image_subdirs:
        
        # Get name of sample:
        name = os.path.basename(image_subdir)
        mask_path = os.path.join(mask_dir, image_subdir, "mask")
        print(f"Processing {name}")
        image_path = os.path.join(image_dir, image_subdir, "result.mha")
        print(f"Image path: {image_path}")
        print(f"Mask path: {mask_path}")
        # Load images
        if os.path.exists(image_path) and os.path.exists(mask_path):
            roi = dict[name.split("_")[0]]

            print(f"Image and Mask exist for {name}")
            image = sitk.ReadImage(image_path)
            image.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
            image.SetOrigin([0, 0, 0])
            image = image[roi[0]:roi[1], roi[2]:roi[3], :]
            print(f"Image size: {image.GetSize()}")
            size_z = [image.GetSize()[2], image.GetSize()[1], image.GetSize()[0]]
            print(f"Size z: {size_z}")
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


        




            