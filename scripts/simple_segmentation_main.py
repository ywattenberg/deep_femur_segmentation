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

torch.set_default_dtype(torch.float32)

def main():
    config = yaml.safe_load(open("config/segmentation_config.yaml", "r"))
    config["context_csv_path"] = r"HRpQCT_aim\\Cropped_regions.csv"
    if "seed" in config:
        torch.manual_seed(config["seed"])
    dataset = FemurSegmentationDataset(config, split="test")

    # for i in range(len(dataset)):
    #     x, y, mask = dataset[i]
    #     fig, ax = plt.subplots(2, 2, dpi=300, figsize=(10, 10))
    #     ax[0, 0].imshow(x[0,0], cmap="gray")
    #     ax[0,0].set_title("Input")
    #     ax[0, 1].imshow(y[0,0], cmap="gray")
    #     ax[0,1].set_title("Hr-pQCT")
    #     ax[1, 0].imshow(mask[0, 0], cmap="gray")
    #     ax[1,0].set_title("Cort")
    #     ax[1, 1].imshow(mask[1, 0], cmap="gray")
    #     ax[1,1].set_title("Trab")
    #     plt.savefig(f"test/sample_{i}.png")

    val_config = config.copy()
    val_config["context_csv_path"] = r"HRpQCT_aim\\numpy\\Cropped_regions_val.csv"
    val_dataset = FemurSegmentationDataset(val_config, split="val")
    model = monai_nets.UNet(
        spatial_dims=config["model"]["spatial_dims"],
        in_channels=1,
        out_channels=2 if config["use_cortical_and_trabecular"] else 1,
        channels=config["model"]["features"],
        strides=config["model"]["strides"],
        dropout=config["model"]["dropout"],
        norm=config["model"]["norm"],
        act=config["model"]["activation"],
    )
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    trainer = Trainer(model, dataset, val_dataset, DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True), optimizer, config)
    trainer.train_test(epochs_between_test=50)

if __name__ == "__main__":
    main()




    
