import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.dataset.dataset_segmentation import FemurSegmentationDataset
from src.dataset.transforms import get_image_segmentation_augmentation

torch.set_default_dtype(torch.float32)

def plot_overview(dataset, n_samples=4):
    fig, ax = plt.subplots(n_samples, 4, dpi=800, figsize=(7, 10))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0.2, top=0.95, bottom=0.05, left=0.1, right=0.9)
    x, y, mask = dataset[0]
    ax[0, 0].imshow(x[0,0], cmap="gray")
    ax[0,0].set_title("PCCT image")
    ax[0,0].axis("off")
    ax[0,0].set_ylabel("Sample 1")
    ax[0, 1].imshow(y[0,0], cmap="gray")
    ax[0,1].set_title("Hr-pQCT image")
    ax[0,1].axis("off")
    ax[0, 2].imshow(mask[0, 0], cmap="gray")
    ax[0,2].set_title("Cortical mask")
    ax[0,2].axis("off")
    ax[0, 3].imshow(mask[1, 0], cmap="gray")
    ax[0,3].set_title("Trabecular mask")
    ax[0,3].axis("off")
    for i in range(1, n_samples):
        x, y, mask = dataset[i]
        ax[i, 0].imshow(x[0,0], cmap="gray")
        ax[i, 0].axis("off")   
        ax[i, 1].imshow(y[0,0], cmap="gray")
        ax[i, 1].axis("off")
        ax[i, 2].imshow(mask[0, 0], cmap="gray")
        ax[i, 2].axis("off")
        ax[i, 3].imshow(mask[1, 0], cmap="gray")
        ax[i, 3].axis("off")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0, top=0.95, bottom=0.05, left=0.1, right=0.9)
    plt.savefig(f"test/dataset_viz/all.png")

def plt_augmented_images(dataset, config, n_samples=2):
    aug_stack = get_image_segmentation_augmentation(config, "train")
    

    fig, ax = plt.subplots(n_samples, 2, dpi=300, figsize=(10, 10))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0.2, top=0.95, bottom=0.05, left=0.1, right=0.9)
    x, y, mask = dataset[i]
    ax[0,0].imshow(x[0,0], cmap="gray")
    ax[0,0].set_title("PCCT image")
    ax[0,0].axis("off")
    ax[0,1].imshow(y[0,0], cmap="gray")
    for i in range(1, n_samples):
        x, y, mask = dataset[i]
        ax[i,0].imshow(x[0,0], cmap="gray")
        

    fig.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0, top=0.95, bottom=0.05, left=0.1, right=0.9)
    plt.savefig(f"test/dataset_viz/all.png")

def main():
    config = yaml.safe_load(open("config/segmentation_config.yaml", "r"))
    # config["context_csv_path"] = r"HRpQCT_aim\\Cropped_regions.csv"
    if "seed" in config:
        torch.manual_seed(config["seed"])
    dataset = FemurSegmentationDataset(config, split="test")



if __name__ == "__main__":
    main()




    
