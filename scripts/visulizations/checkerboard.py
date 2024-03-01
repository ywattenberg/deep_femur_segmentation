import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import yaml
import os 
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.utils import image_to_array


def main(fixed_image: str, moving_image: str | sitk.Image, number_of_slices: int, number_of_tiles: tuple[int, int]):
    """
    Create a checkerboard image from the fixed and moving image to check the registration quality. This function takes a fixed image and a moving image and creates a checkerboard image from them. The checkerboard image is saved as a png file. The number of slices to use can be specified. The number of tiles in the checkerboard can also be specified. The default is 10 slices and a 15x15 checkerboard.

    -----------------
    Parameters:
    -----------------
    fixed_image: str
        The path to the fixed image.
    moving_image: str | sitk.Image
        The path to the moving image or the moving image itself.
    number_of_slices: int
        The number of slices to use.
    number_of_tiles: tuple[int, int]
        The number of tiles in the checkerboard.
    """
    if isinstance(moving_image, str):
        moving_image = sitk.ReadImage(moving_image)

    parts = [fixed_image + "/part_" + str(i) for i in range(3)]
    slices = []
    for part in parts:
        slices_in_folder = os.listdir(part)
        slices_in_folder.sort()
        for slice in slices_in_folder:
            slices.append(os.path.join(part, slice))

    print(f"Number of slices: {len(slices)} in fixed image.")
    print(f"Number of slices: {moving_image.GetDepth()} in moving image.")
    m_space_between_slices = moving_image.GetDepth() // number_of_slices
    f_space_between_slices = len(slices) // number_of_slices
    print(f"Space between slices: {m_space_between_slices} in moving image.")
    print(f"Space between slices: {f_space_between_slices} in fixed image.")
    moving_image = moving_image[:,:,::m_space_between_slices]
    moving_image = image_to_array(moving_image)

    m_mean = np.mean(moving_image)
    m_std = np.std(moving_image)
    moving_image = (moving_image - m_mean) / m_std


    fixed_slice = sitk.ReadImage(slices[0])
    fixed_slices = [sitk.GetArrayFromImage(fixed_slice)]
    for i in range(f_space_between_slices, len(slices), f_space_between_slices):
        print(f"Reading slice {i}")
        fixed_slice = sitk.GetArrayFromImage(sitk.ReadImage(slices[i]))
        # fixed_slice = (fixed_slice - f_mean) / f_std
        fixed_slices.append(fixed_slice)

    # Stack the slices into a 3D image
    fixed_slices = [np.squeeze(slice) for slice in fixed_slices]
    fixed_image = np.stack(fixed_slices, axis=2)
    f_mean = np.mean(fixed_image)
    f_std = np.std(fixed_image)
    fixed_image = (fixed_image - f_mean) / f_std
    print(f"Fixed image mean {f_mean} std {f_std}")
    print(f"Fixed image min {np.min(fixed_image)} max {np.max(fixed_image)}")
    print(f"Moving image min {np.min(moving_image)} max {np.max(moving_image)}")

    mask = np.zeros(fixed_image.shape[:2], dtype=np.uint8)
    tile_size = (fixed_image.shape[0] // number_of_tiles[0], fixed_image.shape[1] // number_of_tiles[1])
    for i in range(number_of_tiles[0]):
        for j in range(number_of_tiles[1]):
            if (i + j) % 2 == 0:
                mask[i * tile_size[0]:(i + 1) * tile_size[0], j * tile_size[1]:(j + 1) * tile_size[1]] = 1
    inverse_mask = 1 - mask

    # For each slice, create apply the checkerboard mask
    for i in range(min(fixed_image.shape[2], moving_image.shape[2])):
        up_sampled = cv2.resize(moving_image[:,:,i], dsize=(fixed_image.shape[0], fixed_image.shape[1]))
        fixed_image[:,:,i] *= mask
        up_sampled *= inverse_mask
        # Merge the images and save each slice as a png
        plt.imsave(f"data//checkerboard_{i}.png", fixed_image[:,:,i] + up_sampled, cmap="gray")
    



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