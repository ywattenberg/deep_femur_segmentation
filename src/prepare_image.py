import argparse
import SimpleITK as sitk
import yaml

from preprocessing.preprocessing import safe_image_to_np_series
from preprocessing.calcualte_statistics import calculate_statistics

if __name__ == "__main__":
    # Parse path to image and output path
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the image to convert.")
    parser.add_argument("output", help="Path to the output files.")
    args = parser.parse_args()
    image = sitk.ReadImage(args.input)
    stats = calculate_statistics(image)
    print(stats)
    # Safe the statistics
    with open(args.output + "/statistics.yml", "w") as file:
        yaml.dump(stats, file)

    # Convert the image
    safe_image_to_np_series(image, args.output + "/np_series", roi=[300, 800, 200, 900] )