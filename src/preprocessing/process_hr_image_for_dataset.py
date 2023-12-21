import os
import SimpleITK as sitk
import numpy as np
import argparse
import yaml
import io
import zipfile
import matplotlib.pyplot as plt
from .utils import read_dicom_series, image_to_array, array_to_image

def main(image_folder_path, roi):
    # ROI is from downsampled image thus we need to upscale it
    roi = [r*2 for r in roi]
    print(f"Using ROI: {roi}")
    name = image_folder_path.split("\\")[-1]
    print(f"Processing {name}")
    folders = os.listdir(image_folder_path)
    folders.sort()
    folders = [folder for folder in folders if folder.startswith("part_")]
    folders = [os.path.join(image_folder_path, folder) for folder in folders if os.path.isdir(os.path.join(image_folder_path, folder))]
    print(f"Found the following folders: {folders}")

    assert len(folders) > 0, "No folders found"

    means, stds, vols = [], [], []
    
    for folder in folders:
        image = read_dicom_series(folder)
        image = image_to_array(image)[roi[2]:roi[3], roi[0]:roi[1]]
        print(f"Image shape: {image.shape}")
        for i in range(0, image.shape[2], 500):
            plt.imshow(image[:, :, i])
            plt.savefig(f"{folder}_image_{i}.png")
        means.append(np.mean(image))
        stds.append(np.std(image))
        vols.append(np.prod(image.shape))

    overall_mean = np.float64(np.sum([m*v for m, v in zip(means, vols)])/np.sum(vols))
    # Compute biased estimate of variance (i.e. divide by N, not N-1) see link below
    # We divide ESS and GGS immediately by the sum of the volumes, for numerical stability
    # https://sealevel.info/climate/composite_standard_deviations.html
    total_volume = np.sum(vols, dtype=np.float64)
    ess = np.sum([np.float64(std)**2 * (vol/total_volume) for std, vol in zip(stds, vols)])
    print(f"ESS: {ess}")
    tgss = np.sum([((np.float64(m) - overall_mean)**2) * (v/total_volume) for m, v in zip(means, vols)])
    print(f"TGSS: {tgss}")
    overall_std = np.sqrt(ess + tgss)
    print(f"Overall mean: {overall_mean}, overall std: {overall_std}")

    # Save the overall mean and std
    with open(os.path.join(image_folder_path, "stats.yaml"), "w") as f:
        yaml.dump({"name": name, "mean": float(overall_mean), "std": float(overall_std)}, f)
    
    # # load overall mean and std
    # with open(os.path.join(image_folder_path, "stats.yaml"), "r") as f:
    #    stats = yaml.load(f, Loader=yaml.FullLoader)
    # overall_mean = stats["mean"]
    # overall_std = stats["std"] 

    min, max = [], []
    p99_5, p00_5 = [], []
    # Create empty npz

    offset = 0
    for folder in folders:
        image = read_dicom_series(folder)
        image = image_to_array(image)[roi[2]:roi[3], roi[0]:roi[1]]
        print(f"Image shape: {image.shape}")
        image = (image - overall_mean) / overall_std
        min.append(np.min(image))
        max.append(np.max(image))
        p99_5.append(np.percentile(image, 99.5))
        p00_5.append(np.percentile(image, 0.5))
        
        with zipfile.ZipFile(os.path.join(image_folder_path, f"{name}.zip"), "a", zipfile.ZIP_DEFLATED) as f:
            for slice in range(image.shape[2]):
                io_buffer = io.BytesIO()
                index = slice + offset
                print(f"Saving slice {index}")
                np.save(io_buffer, image[:, :, slice])
                f.writestr(f"{index}.npy", io_buffer.getvalue())
                io_buffer.close()
        
        offset += image.shape[2]
    
    # Save the overall min, max, p99.5 and p00.5
    overall_min = np.min(min)
    overall_max = np.max(max)
    overall_p99_5 = np.max(p99_5)
    overall_p00_5 = np.min(p00_5)

    with open(os.path.join(image_folder_path, "stats.yaml"), "a") as f:
        yaml.dump({"min": float(overall_min), "max": float(overall_max), "p99_5": float(overall_p99_5), "p00_5": float(overall_p00_5)}, f)


if __name__ == "__main__":
    image_folder_path = r"data\HR-pQCT"

    folders = os.listdir(image_folder_path)
    
    dict = {"1":[395, 835, 320, 1000],"3": [360,800,250,1010], "5":[315,850,375,955], "7":[330,775,285,970], "8":[330, 855, 265,1080], "12":[360,825,230,1000]}
    dict = {"8":[330, 855, 265,1080]}
    
    for key, roi in dict.items():
        sample = [folder for folder in folders if folder.startswith(f"{key}_")]
        if len(sample) > 0:
            sample = sample[0]
            main(os.path.join(image_folder_path, sample), roi)
        
        
        


