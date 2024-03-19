import torch
import yaml
import monai.networks.nets as monai_nets
from monai.losses import DiceLoss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer_seg import Trainer
from src.dataset.dataset_segmentation import FemurSegmentationDataset
from src.model.retina_UNet import Retina_UNet
from src.loss.DiceL1Loss import DiceL1Loss

torch.set_default_dtype(torch.float32)

def main():
    config = yaml.safe_load(open("config/segmentation_config.yaml", "r"))
    # config["context_csv_path"] = r"HRpQCT_aim\\Cropped_regions.csv"
    if "seed" in config:
        torch.manual_seed(config["seed"])
    dataset = FemurSegmentationDataset(config, split="test")
    plt.axis("off")

    fig, ax = plt.subplots(4, 4, dpi=800, figsize=(7, 10))
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
    for i in range(1, 4):
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

    return

    val_config = config.copy()
    val_config["context_csv_path"] = r"HRpQCT_aim\\numpy\\Cropped_regions_val.csv"
    val_dataset = FemurSegmentationDataset(val_config, split="val")
    model = monai_nets.BasicUNet(
        spatial_dims=config["model"]["spatial_dims"],
        in_channels=1,
        out_channels=2 if config["use_cortical_and_trabecular"] else 1,
        features=config["model"]["features"],
        #strides=config["model"]["strides"],
        dropout=config["model"]["dropout"],
        norm=config["model"]["norm"],
        act=config["model"]["activation"],
    )
    # model = Retina_UNet(1, 2, 1, config)
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    loss_fn = DiceL1Loss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    trainer = Trainer(model, dataset, val_dataset, loss_fn, optimizer, config)
    trainer.train_test(epochs_between_test=10)

if __name__ == "__main__":
    main()




    
