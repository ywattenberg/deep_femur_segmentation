import argparse
import SimpleITK as sitk
import yaml
import numpy as np

from preprocessing.preprocessing import safe_image_to_np_series
from preprocessing.calcualte_statistics import calculate_statistics
from preprocessing.utils import image_to_array

if __name__ == "__main__":
    # Parse path to image and output path
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the image to convert.")
    parser.add_argument("output", help="Path to the output files.")

    args = parser.parse_args()
    image = image_to_array(sitk.ReadImage(args.input))
    original_min = np.min(image)
    original_max = np.max(image)

    # shift image to [0, 1] range
    image = (image - min) / (max - min)

    stats = calculate_statistics(image)

    #safe original stats for denomalization if neccessary
    stats["original_min"] = original_min
    stats["original_max"] = original_max

    print(stats)
    # Safe the statistics
    with open(args.output + "/statistics.yml", "w") as file:
        yaml.dump(stats, file)

    # Convert the image
    safe_image_to_np_series(image, args.output + "/np_series" )