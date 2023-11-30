import argparse
import SimpleITK as sitk
import yaml
import numpy as np
import os

from preprocessing.preprocessing import safe_image_to_np_series
from preprocessing.calculate_statistics import calculate_statistics
from preprocessing.utils import image_to_array

def perpare_HR_pQCT(input, output, roi):
    # As the images are to large to fit into memory, we convert them in three parts
    paths = [folder for folder in os.listdir(input) if folder.startswith("part")]
    paths = [os.path.join(input, folder) for folder in paths]

    # Load statistics
    with open(input + "/statistics.yml", "r") as file:
        overall_stats = yaml.load(file, Loader=yaml.FullLoader)

    std = overall_stats["original_std"]
    mean = overall_stats["original_mean"]

    index = 0
    for folder in paths:
        # Read DICOM series
        image_reader = sitk.ImageSeriesReader().SetFileNames(sitk.ImageSeriesReader().GetGDCMSeriesFileNames(folder))
        image = sitk.GetArrayFromImage(image_reader.Execute())
        image = image[:, roi[0]:roi[1], roi[2]:roi[3]]
        image = (image - mean) / std
        safe_image_to_np_series(image, output + "/np_series", roi=[300, 800, 200, 900], start_index=index)
        index += image.shape[0]

def prepare_PCCT(input, output, roi):
    image = sitk.GetArrayFromImage(sitk.ReadImage(args.input))
    original_min = np.min(image)
    original_max = np.max(image)

    stats = calculate_statistics(image)

    # normalize image
    image = (image - stats["intensity"]["mean"]) / stats["intensity"]["std"]

    #safe original stats for denomalization if neccessary
    stats["original_min"] = original_min
    stats["original_max"] = original_max

    print(stats)
    # Safe the statistics
    with open(args.output + "/statistics.yml", "w") as file:
        yaml.dump(stats, file)

        # Convert the image
    safe_image_to_np_series(image, args.output + "/np_series", roi=[300, 800, 200, 900] )


if __name__ == "__main__":
    # Parse path to image and output path
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the image to convert.")
    parser.add_argument("output", help="Path to the output files.")

    args = parser.parse_args()
    
    if "HR-pQCT" in args.input:
        perpare_HR_pQCT(args.input, args.output, roi=[150, 1600, 100, 1800])
    elif "PCCT" in args.input:
        prepare_PCCT(args.input, args.output, roi=[300, 800, 200, 900])
    else:
        raise ValueError("Unknown dataset. Please specify either HR-pQCT or PCCT.")

