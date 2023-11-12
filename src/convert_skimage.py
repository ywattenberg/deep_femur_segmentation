import argparse

from preprocessing.preprocessing import safe_image_to_np_series

if __name__ == "__main__":
    # Parse path to image and output path
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to the image to convert.")
    parser.add_argument("output", help="Path to the output file.")
    args = parser.parse_args()

    # Convert the image
    safe_image_to_np_series(args.input, args.output)