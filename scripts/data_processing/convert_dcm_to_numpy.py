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
import yaml

def read_dicom_series(file_list):
    np_array = []
    for file in file_list:
        image = pydicom.dcmread(file)
        # print(image.ImagePositionPatient)
        np_array.append(image.pixel_array)
    return np.array(np_array)

def print_tags(file_list):
    file = file_list[0]
    image = pydicom.dcmread(file)
    print(image)

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

    # Convert all each dicom file to numpy array
    files = os.listdir(os.path.join(in_folder, folder))
    files = [file for file in files if file.endswith(".dcm")]

    image_files = [file for file in files if len(file.split("_")) == 2]
    seg_files = [file for file in files if "seg" in file]
    outer_files = [file for file in files if "outer" in file]
    cortical_files = [file for file in files if "cort" in file]
    trabecular_files = [file for file in files if "trab" in file]
       
    outer = read_dicom_series([os.path.join(in_folder, folder, file) for file in outer_files])
    outer_mask = np.zeros_like(outer, dtype=bool)
    outer_mask[outer > 0] = True

    # Read metadata from outer mask
    try:
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(os.path.join(in_folder, folder))
        files = [file for file in files if "outer" in file]
        reader.SetFileNames(files)
        image = reader.Execute()
    except:
        print(f"Could not read metadata from {folder}")
        with open("log.log", 'a') as f:
            f.write(f"Could not read metadata from {folder}\n")
        return
    print(f"Origin: {image.GetOrigin()}")
    print(f"Spacing: {image.GetSpacing()}")
    return
    if os.path.exists(os.path.join(output_folder, folder, "image_stats.yaml")):
        image_stats = yaml.load(open(os.path.join(output_folder, folder, "image_stats.yaml"), "r"), Loader=yaml.FullLoader)
    else:
        image_stats = {}
    spacing = list(image.GetSpacing())
    origin = list(image.GetOrigin())
    offset = [a/b for a,b in zip(origin, spacing)]
    print(f"Offset: {offset}")
    offset = [int(np.round(i)) for i in offset]
    offset = [offset[1], offset[0]]
    image_stats["outer_offset"] = offset
    print(image_stats)
    with open(os.path.join(output_folder, folder, "image_stats.yaml"), "w") as f:
        yaml.dump(image_stats, f)
    

    # outer_mask = outer_mask[: , :, :]
    # print(f"Outer mask shape: {outer_mask.shape}")
    # ax = plt.subplots(1, 6)[1]
    # ax[0].imshow(outer_mask[0, :, :])
    # ax[1].imshow(outer_mask[1, :, :])
    # ax[2].imshow(outer_mask[2, :, :])
    # ax[3].imshow(outer_mask[-1, :, :])
    # ax[4].imshow(outer_mask[-2, :, :])
    # ax[5].imshow(outer_mask[-3, :, :])
    # plt.show()

    # np.save(os.path.join(output_folder, folder, f"{sample_name}_outer_dicom.npy"), outer_mask)


def main(in_folder, output_folder):
    # Go through all folders in folder and convert the dicom files to numpy arrays
    folders = os.listdir(in_folder)
    folders = ["flowbone_2210_06147_l__m", "flowbone_2208_04835_r__p"]
    df = pd.read_csv(r"C:\Users\Yannick\Documents\repos\deep_femur_segmentation\data\HRpQCT_annotated\Cropped_regions.csv", delimiter=";")
    for folder in folders:
        print(folder)
        process_folder(folder, in_folder, output_folder, df)


    # with mp.Pool(8) as pool:
    #     pool.starmap(process_folder, [(folder, in_folder, output_folder, df) for folder in folders])
    







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--folder", type=str, required=True)
    parser.add_argument("-o","--output_folder", type=str, required=True)
    args = parser.parse_args()
    main(args.folder, args.output_folder)

        
