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
        print(os.path.join(path, title + ".png"))
        plt.savefig(os.path.join(path, title + ".png"))
    else:
        plt.show()
    plt.close()

def threshold_image(image, threshold, sigma=[1, 1, 1]):
    #image = gaussian(image, sigma=sigma)
    mask = image > threshold
    return mask

def main(data_folder, out_folder, threshold=0.6, factor=1, visualize=False):
    files = os.listdir(data_folder)
    files = sorted([file for file in files if file.endswith(".npy")])
    print(f"Found {len(files)} files")
    if visualize:
        image = []
        for i in range(0, len(files), 100):
            image.append(np.load(os.path.join(data_folder, files[i])))
        print(f"Loaded {len(image)} images")
        image = np.stack(image)
        print(f"Min: {image.min()}, Max: {image.max()}, Mean: {image.mean()}, Std: {image.astype(np.float64).std()}")
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

def it_subdirs(folders):
    for folder in os.listdir(folders):
        path = os.path.join(folders, folder, folder)
        if os.path.isdir(path):
            yield path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", "-d", type=str, required=True)
    parser.add_argument("--out_folder", "-o", type=str, required=True)
    parser.add_argument("--threshold", "-t", type=float, default=0.6)
    parser.add_argument("--factor", "-f", type=int, default=1)
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--subdirs", "-s", action="store_true")
    args = parser.parse_args()
    if args.subdirs:
        for folder in it_subdirs(args.data_folder):
            print(f"Processing folder {folder}")
            os.makedirs(os.path.join(folder, "../mask"), exist_ok=True)
            main(folder, os.path.join(folder, "../mask"), args.threshold, args.factor, args.visualize)
    else:
        main(args.data_folder, args.out_folder, args.threshold, args.factor, args.visualize)
    