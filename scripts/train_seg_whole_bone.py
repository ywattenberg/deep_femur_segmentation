import torch
import yaml
import monai.networks.nets as monai_nets
from monai.losses import DiceLoss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer.trainer import Trainer
from src.dataset.dataset_segmentation_full import FemurImageSegmentationDataset 
from src.utils.dtypes import TORCH_DTYPES, NUMPY_DTYPES



def main():

    # Get Unet model from Monai
    CONFIG = yaml.safe_load(open("config/config.yaml", "r"))
    if "seed" in CONFIG:
        torch.manual_seed(CONFIG["seed"])
    print(CONFIG)
    d_type = TORCH_DTYPES[CONFIG["dtype"]]
    torch.set_default_dtype(d_type)
    config = CONFIG
    dataset = FemurImageSegmentationDataset(config=CONFIG, split="train")
    val_conf = CONFIG.copy()
    val_conf["context_csv_path"] = "data/validation.csv"
    val_dataset = FemurImageSegmentationDataset(config=val_conf, split="val")
    model = monai_nets.UNet(
        spatial_dims=config["model"]["spatial_dims"],
        in_channels=1,
        out_channels=1,
        channels=config["model"]["features"],
        strides=config["model"]["strides"],
        dropout=config["model"]["dropout"],
        norm=config["model"]["norm"],
        act=config["model"]["activation"],
    )

    # Load model
    # model.load_state_dict(torch.load("models/BasicUNet_epoch_5.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    loss = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    trainer = Trainer(model, dataset, val_dataset, loss, optimizer, CONFIG, test_data=val_dataset)
    trainer.train_test()
    
        
   



if __name__ == "__main__":
    main()
