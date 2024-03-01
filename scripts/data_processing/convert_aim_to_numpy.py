import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom 
import argparse
import yaml
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
import itk

def itk_sitk(image):
    sitk_image = sitk.GetImageFromArray(
        itk.GetArrayFromImage(image),
        isVector=image.GetNumberOfComponentsPerPixel() > 1,
    )
    sitk_image.SetOrigin(tuple(image.GetOrigin()))
    sitk_image.SetSpacing(tuple(image.GetSpacing()))
    sitk_image.SetDirection(itk.GetArrayFromMatrix(image.GetDirection()).flatten())
    return sitk_image

def swap_xy(image):
    original_spacing = image.GetSpacing()
    original_origin = image.GetOrigin()
    new_origin = [original_origin[1], original_origin[0], original_origin[2]]
    new_spacing = [original_spacing[1], original_spacing[0], original_spacing[2]]

    image.SetOrigin(new_origin)
    image.SetSpacing(new_spacing)
    return image

def read_aim(path, dtype="signed short"):
    image_type = itk.Image[itk.ctype(dtype), 3]
    reader = itk.ImageFileReader[image_type].New()
    image_io = itk.ScancoImageIO.New()
    reader.SetImageIO(image_io)
    reader.SetFileName(path)
    reader.Update()
    img = reader.GetOutput()
    # print(np.sum(itk.GetArrayFromImage(img)))
    return itk_sitk(img)

def calculate_offset(image):
    offset = [image.GetOrigin()[i]/image.GetSpacing()[i] for i in range(2)]
    offset = [int(np.ceil(offset[i])) for i in range(2)]
    return offset

def proccess_mask(image):
    # if depth is 151 cut first slice
    image = image[:, :, 1:]
    image = swap_xy(image)
    offset = calculate_offset(image)
    img_array = sitk.GetArrayFromImage(image)
    mask = np.zeros_like(img_array, dtype=bool)
    mask[img_array > 64] = True
    return mask, offset


def process_folder(folder, in_folder, output_folder, df):
    print(f"Processing {folder}")
    print(f"Create output folder {os.path.join(output_folder, folder)}")
    os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
    curr_row = None

    for row in df.iterrows():
        sample_name = row[1]["SampName"].lower().replace(" ", "_")
        sample_name = sample_name + '__' + row[1]["Loc"].lower()
        if folder.lower() == sample_name:
            print(f"Found {folder} in context")
            curr_row = row
            break

    if curr_row is None:
        print(f"Could not find {folder} in context")
        return
    

    sample_name = folder.split("_")[1:3]
    sample_name = "_".join(sample_name)

    # Convert .aim files to numpy arrays
    files = os.listdir(os.path.join(in_folder, folder))
    files = [file for file in files if file.endswith(".aim")]

    image_files = [file for file in files if len(file.split("_")) == 1]
    seg_files = [file for file in files if "seg" in file]
    outer_files = [file for file in files if "outer" in file]
    cortical_files = [file for file in files if "cort" in file]
    trabecular_files = [file for file in files if "trab" in file]

    image = read_aim(os.path.join(in_folder, folder, image_files[0]), dtype="signed short")
    image = sitk.GetArrayFromImage(image).astype(np.float64)
    image_stats = { "name": sample_name, "mean": float(np.mean(image)), "std": float(np.std(image)), "min": float(np.min(image)), "max": float(np.max(image)), "shape": list(image.shape)}

    image = (image - image_stats["mean"])/image_stats["std"]
    image = image.astype(np.float16)
    # print(image.shape)

    if folder.endswith("p") or folder.endswith("m"):
        trabecular = read_aim(os.path.join(in_folder, folder, trabecular_files[0]), dtype="unsigned char")
        trabecular_mask, offset = proccess_mask(trabecular)
        image_stats["trabecular_offset"] = list(offset)


        cortical = read_aim(os.path.join(in_folder, folder, cortical_files[0]), dtype="unsigned char")
        cortical_mask, offset = proccess_mask(cortical)
        image_stats["cortical_offset"] = list(offset)
    


    outer = read_aim(os.path.join(in_folder, folder, outer_files[0]), dtype="unsigned char")
    # print(outer.GetSize())
    outer_mask, offset = proccess_mask(outer)

    image_stats["outer_offset"] = list(offset)
    with open(os.path.join(output_folder, folder, "image_stats.yaml"), "w") as f:
        yaml.dump(image_stats, f)

    
    # image = image[0:min_shape[0], min_loc[1]:min_shape[1]+min_loc[1], min_loc[2]:min_shape[2]+min_loc[2]]
    if folder.endswith("p") or folder.endswith("m"):
        # fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        # ax[0].imshow(image[0, :, :])
        # ax[0].set_title("Image")
        # ax[1].imshow(outer_mask[0, :, :])
        # ax[1].set_title("Outer mask")
        # ax[2].imshow(cortical_mask[0, :, :])
        # ax[2].set_title("Cortical mask")
        # ax[3].imshow(trabecular_mask[0, :, :])
        # ax[3].set_title("Trabecular mask")
        # ax[4].imshow(image[0, :, :])
        # offset = image_stats["outer_offset"]
        # print(offset)
        # print(outer_mask.shape)
        # print(image.shape)
        # new_mask = np.zeros_like(image[0, :, :], dtype=bool)
        # new_mask[offset[0]:offset[0]+outer_mask.shape[1], offset[1]:offset[1]+outer_mask.shape[2]] = outer_mask[0, :, :]
        # ax[4].imshow(new_mask, cmap='gray', alpha=0.5)
        # plt.show()
        np.save(os.path.join(output_folder, folder, f"{sample_name}_cortical.npy"), cortical_mask)
        np.save(os.path.join(output_folder, folder, f"{sample_name}_trabecular.npy"), trabecular_mask)
    np.save(os.path.join(output_folder, folder, f"{sample_name}_image.npy"), image)
    np.save(os.path.join(output_folder, folder, f"{sample_name}_outer.npy"), outer_mask)


def main(in_folder, output_folder):
    # Go through all folders in folder and convert the dicom files to numpy arrays
    folders = os.listdir(in_folder)
    df = pd.read_csv(r"C:\Users\Yannick\Documents\repos\deep_femur_segmentation\data\HRpQCT_aim\Cropped_regions.csv", delimiter=";")
    # folders.reverse()
    # for folder in folders:
    #     print(folder)
    #     process_folder(folder, in_folder, output_folder, df)


    with mp.Pool(4) as pool:
        pool.starmap(process_folder, [(folder, in_folder, output_folder, df) for folder in folders])
    







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--folder", type=str, required=True)
    parser.add_argument("-o","--output_folder", type=str, required=True)
    args = parser.parse_args()
    main(args.folder, args.output_folder)

        
