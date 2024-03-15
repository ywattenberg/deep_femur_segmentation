import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu, threshold_otsu, gaussian, threshold_triangle
import skimage



def create_mask(image):
    """
    Create a mask from the input image.
    Args:
        image: A numpy array with shape (H, W, D)
    Returns:
        A numpy array with shape (H, W, D)
    """
    # Create a mask from the input image
    image = gaussian(image, sigma=1)
    image = skimage.exposure.rescale_intensity(image, in_range=(-1, 4), out_range=(0, 1))
    binary_image = image > 0.4
    return binary_image

def main(path_to_dir, save_path):
    for dir in os.listdir(path_to_dir):
        if not os.path.isdir(os.path.join(path_to_dir, dir)):
            continue
        for file in reversed(os.listdir(os.path.join(path_to_dir, dir))):
            if "image" in file and file.endswith(".npy"):
                print(f"Creating mask for {file}")
                image = np.load(os.path.join(path_to_dir, dir, file))
                mask = create_mask(image)
                # fig, axs = plt.subplots(1, 2, figsize=(10, 10))
                # axs[0].imshow(image[0, :, :], cmap="gray")
                # axs[0].set_title("Input")
                # axs[1].imshow(mask[0, :, :], cmap="gray")
                # axs[1].set_title("Mask")
                # plt.savefig(os.path.join(save_path, file.replace("image", "mask").replace(".npy", ".png")))
                np.save(os.path.join(path_to_dir, dir, file.replace("image", "threshold")), mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dir", "-p", type=str, help="Path to the directory containing the images")
    parser.add_argument("--save_path", "-s", type=str, help="Path to the directory where the masks should be saved")
    args = parser.parse_args()
    main(args.path_to_dir, args.save_path)



    