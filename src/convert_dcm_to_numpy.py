import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom 
import argparse
import yaml
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd

def read_dicom_series(file_list):
    np_array = []
    for file in file_list:
        image = pydicom.dcmread(file)
        np_array.append(image.pixel_array)
    return np.array(np_array)

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
    

    min_loc = [int(curr_row[1]["MinZ"]), int(curr_row[1]["MinX"]), int(curr_row[1]["MinY"])]
    print(f"Min loc: {min_loc}")
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


    image = read_dicom_series([os.path.join(in_folder, folder, file) for file in image_files]).astype(np.float64)
    # image_stats = { "name": sample_name, "mean": float(np.mean(image)), "std": float(np.std(image)), "min": float(np.min(image)), "max": float(np.max(image)), "shape": list(image.shape)}
    # with open(os.path.join(output_folder, folder, "image_stats.yaml"), "w") as f:
    #     yaml.dump(image_stats, f)

    # image = (image - image_stats["mean"])/image_stats["std"]
    image = image.astype(np.float16)

    seg = read_dicom_series([os.path.join(in_folder, folder, file) for file in seg_files]).astype(np.float64)
    # seg = (seg - image_stats["mean"])/image_stats["std"]
    seg = seg.astype(np.float16)

    outer = read_dicom_series([os.path.join(in_folder, folder, file) for file in outer_files])
    outer_mask = np.zeros_like(outer, dtype=bool)
    outer_mask[outer > 0] = True

    imgs = [seg, outer_mask]

    if folder.endswith("p") or folder.endswith("m"):
        cortical = read_dicom_series([os.path.join(in_folder, folder, file) for file in cortical_files])
        cortical_mask = np.zeros_like(cortical, dtype=bool)
        cortical_mask[cortical > 0] = True

        trabecular = read_dicom_series([os.path.join(in_folder, folder, file) for file in trabecular_files])
        trabecular_mask = np.zeros_like(trabecular, dtype=bool)
        trabecular_mask[trabecular > 0] = True

        imgs.extend([cortical_mask, trabecular_mask])
    
    min_shape = list(image.shape)
    for i in imgs:
        for j in range(3):
            min_shape[j] = min(min_shape[j], i.shape[j])

    # image needs to be cropped to the correct size
    # Assume that the masks are centred in the image
    # this means we want to crop the image to the same size as the masks
    image_center = [int(image.shape[0]/2), int(image.shape[1]/2), int(image.shape[2]/2)]
    # Cropping the image to the same size as the masks
    image = image[:, image_center[1]-int(min_shape[1]/2):image_center[1]+int(min_shape[1]/2), image_center[2]-int(min_shape[2]/2):image_center[2]+int(min_shape[2]/2)]
    
    # image = image[0:min_shape[0], min_loc[1]:min_shape[1]+min_loc[1], min_loc[2]:min_shape[2]+min_loc[2]]
    seg = seg[0:min_shape[0], 0:min_shape[1], 0:min_shape[2]]
    outer_mask = outer_mask[0:min_shape[0], 0:min_shape[1], 0:min_shape[2]]
    if folder.endswith("p") or folder.endswith("m"):
        cortical_mask = cortical_mask[0:min_shape[0], 0:min_shape[1], 0:min_shape[2]]
        trabecular_mask = trabecular_mask[0:min_shape[0], 0:min_shape[1], 0:min_shape[2]]
    print(f"Image shape: {image.shape}")
    print(f"Seg shape: {seg.shape}")
    print(f"Outer shape: {outer_mask.shape}")
    if folder.endswith("p") or folder.endswith("m"):
        print(f"Cortical shape: {cortical_mask.shape}")
        print(f"Trabecular shape: {trabecular_mask.shape}")
    # np.save(os.path.join(output_folder, folder, f"{sample_name}_image.npy"), image)
    # np.save(os.path.join(output_folder, folder, f"{sample_name}_outer.npy"), outer_mask)
    # np.save(os.path.join(output_folder, folder, f"{sample_name}_cortical.npy"), cortical_mask)
    # np.save(os.path.join(output_folder, folder, f"{sample_name}_trabecular.npy"), trabecular_mask)

    axs, foo = plt.subplots(2,3)
    half_depth = int(image.shape[0]/2)
    foo[0,0].imshow(image[half_depth,:,:])
    foo[0,0].set_title("image")
    foo[0,1].imshow(seg[half_depth,:,:])
    foo[0,1].set_title("seg")
    foo[0,2].imshow(outer_mask[half_depth,:,:])
    foo[0,2].set_title("outer")
    if folder.endswith("p") or folder.endswith("m"):
        foo[1,0].imshow(cortical_mask[half_depth,:,:])
        foo[1,0].set_title("cortical")
        foo[1,1].imshow(trabecular_mask[half_depth,:,:])
        foo[1,1].set_title("trabecular")
    plt.show()


def main(in_folder, output_folder):
    # Go through all folders in folder and convert the dicom files to numpy arrays
    folders = os.listdir(in_folder)
    df = pd.read_csv(r"C:\Users\Yannick\Documents\repos\deep_femur_segmentation\data\HRpQCT_annotated\Cropped_regions.csv", delimiter=";")
    for folder in folders:
        print(folder)
        process_folder(folder, in_folder, output_folder, df)


    # with mp.Pool(1) as pool:
    #     pool.starmap(process_folder, [(folder, in_folder, output_folder, df) for folder in folders])
    







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--folder", type=str, required=True)
    parser.add_argument("-o","--output_folder", type=str, required=True)
    args = parser.parse_args()
    main(args.folder, args.output_folder)

        
