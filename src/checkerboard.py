import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2

from preprocessing.utils import image_to_array
from preprocessing.preprocessing import intensity_normalization

def main(fixed_image: str | sitk.Image, moving_image: str | sitk.Image, number_of_slices: int, number_of_tiles: tuple[int, int]):
    if isinstance(fixed_image, str):
        fixed_image = sitk.ReadImage(fixed_image)
    if isinstance(moving_image, str):
        moving_image = sitk.ReadImage(moving_image)

    space_between_slices = fixed_image.GetDepth() // number_of_slices
    fixed_image = image_to_array(fixed_image[:,:,::space_between_slices])
    moving_image = image_to_array(moving_image[:,:,::space_between_slices])
    moving_image = intensity_normalization(moving_image, 0, 255)
    fixed_image = intensity_normalization(fixed_image, 0, 255)
    
    #fixed_image = intensity_normalization(fixed_image, 0, 255)

    # min = np.min(fixed_image)
    # max = np.max(fixed_image)
    # lower_bound = 75
    # upper_bound = 355

    # LUT = np.zeros(1000, dtype=np.float32)
    # LUT[:lower_bound] = 0
    # LUT[lower_bound:upper_bound] = np.linspace(0, 255, upper_bound - lower_bound)
    # LUT[upper_bound:] = 255
    # fixed_image = LUT[fixed_image.astype(np.uint8)]

    # print(f"Fixed image range: {np.min(fixed_image)}, {np.max(fixed_image)}")
    # print(f"Moving image range: {np.min(moving_image)}, {np.max(moving_image)}")

    # Rescale intensities of fixed ima
    # Generate the checkerboard mask
    mask = np.zeros(fixed_image.shape[:2], dtype=np.uint8)
    tile_size = (fixed_image.shape[0] // number_of_tiles[0], fixed_image.shape[1] // number_of_tiles[1])
    for i in range(number_of_tiles[0]):
        for j in range(number_of_tiles[1]):
            if (i + j) % 2 == 0:
                mask[i * tile_size[0]:(i + 1) * tile_size[0], j * tile_size[1]:(j + 1) * tile_size[1]] = 1
    inverse_mask = 1 - mask

    # For each slice, create apply the checkerboard mask
    for i in range(fixed_image.shape[2]):
        fixed_image[:,:,i] *= mask
        moving_image[:,:,i] *= inverse_mask
        # Merge the images and save each slice as a png
        plt.imsave(f"data/checkerboard_{i}.png", fixed_image[:,:,i] + moving_image[:,:,i], cmap="gray")
    



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("fixed_image", help="The fixed image.")
    parser.add_argument("moving_image", help="The moving image.")
    parser.add_argument("--number_of_slices", help="The number of slices to use.", type=int, default=10)
    parser.add_argument("--grid", help="The grid size.", type=int, nargs=2, default=(15, 15))
    args = parser.parse_args()

    # Run the main function
    main(args.fixed_image, args.moving_image, args.number_of_slices, args.grid)