import os
import sys
import numpy as np
import argparse
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def plt_image(image, mask, title, path=None):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image, cmap="gray")
    axs[1].imshow(mask, cmap="gray")
    plt.suptitle(title)
    if path is not None:
        plt.savefig(os.path.join(path, title + ".png"))
    else:
        plt.show()

def threshold_image(image, threshold, sigma=[1, 1, 1]):
    image = gaussian(image, sigma=sigma)
    mask = image > threshold
    return mask

def main(data_folder, out_folder, threshold=0.7, factor=1, visualize=False):
    files = os.listdir(data_folder)
    files = sorted([file for file in files if file.endswith(".npy")])
    if visualize:
        image = []
        for i in range(len(files), 100):
            image.append(np.load(os.path.join(data_folder, files[i])))
        image = np.stack(image)
        mask = threshold_image(image, threshold)
        for i in range(image.shape[0]):
            plt_image(image[i], mask[i], f"Image {i}", out_folder)
    else:
        image = []
        for i in range(len(files)):
            image.append(np.load(os.path.join(data_folder, files[i])))
        image = np.stack(image)
        mask = threshold_image(image, threshold)
        for i in range(image.shape[0]):
            np.save(os.path.join(out_folder, files[i]), mask[i])

