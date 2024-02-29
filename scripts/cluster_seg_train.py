import torch
import yaml
import monai.networks.nets as monai_nets
from monai.losses import DiceLoss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer_seg import Trainer
from src.dataset.dataset_segmentation import FemurSegmentationDataset 

torch.set_default_dtype(torch.float32)

def main(config_path, epochs_between_test):
    config = yaml.safe_load(open(config_path, "r"))
    # config["context_csv_path"] = r"HRpQCT_aim\\Cropped_regions.csv"
    if "seed" in config:
        torch.manual_seed(config["seed"])
    dataset = FemurSegmentationDataset(config, split="train")

    val_config = config.copy()
    val_config["context_csv_path"] = r"numpy/Cropped_regions_val.csv"
    val_dataset = FemurSegmentationDataset(val_config, split="val")
    model = monai_nets.BasicUNetPlusPlus(
        spatial_dims=config["model"]["spatial_dims"],
        in_channels=1,
        out_channels=2 if config["use_cortical_and_trabecular"] else 1,
        features=config["model"]["features"],
        #strides=config["model"]["strides"],
        dropout=config["model"]["dropout"],
        norm=config["model"]["norm"],
        act=config["model"]["activation"],
    )
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    trainer = Trainer(model, dataset, val_dataset, DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True), optimizer, config)
    trainer.train_test(epochs_between_test=epochs_between_test)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="config/segmentation_config.yaml")
    argparser.add_argument("--epochs_between_test", type=int, default=10)
    main()




    
