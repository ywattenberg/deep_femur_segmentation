import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu, threshold_otsu, gaussian



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
    threshold_otsu_value = threshold_otsu(image)
    binary_image = image > threshold_otsu_value
    return binary_image

def main(path_to_dir):
    for dir in os.listdir(path_to_dir):
        if not os.path.isdir(os.path.join(path_to_dir, dir)):
            continue
        for file in os.listdir(os.path.join(path_to_dir, dir)):
            if "image" in file and file.endswith(".npy"):
                print(f"Creating mask for {file}")
                image = np.load(os.path.join(path_to_dir, dir, file))
                image
                mask = create_mask(image)
                # fig, axs = plt.subplots(1, 2, figsize=(10, 10))
                # axs[0].imshow(image[0, :, :], cmap="gray")
                # axs[0].set_title("Input")
                # axs[1].imshow(mask[0, :, :], cmap="gray")
                # axs[1].set_title("Mask")
                # plt.show()
                np.save(os.path.join(path_to_dir, dir, file.replace("image", "threshold")), mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dir", "-p", type=str, help="Path to the directory containing the images")
    args = parser.parse_args()
    main(args.path_to_dir)



    