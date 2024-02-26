import torch
import yaml
import monai.networks.nets as monai_nets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer import Trainer
from src.dataset.dataset import FemurImageDataset 
from src.dataset.transforms import CustomCropRandomd
from src.model.basic_UNet import BasicUNet, UpsampleUNet, UpsampleUNet_new

from src.utils.dtypes import TORCH_DTYPES, NUMPY_DTYPES


def main():

    # Get Unet model from Monai
    CONFIG = yaml.safe_load(open("config/config.yaml", "r"))
    if "seed" in CONFIG:
        torch.manual_seed(CONFIG["seed"])
    print(CONFIG)
    d_type = TORCH_DTYPES[CONFIG["dtype"]]
    torch.set_default_dtype(d_type)
    dataset = FemurImageDataset(config=CONFIG, split="train")
    val_conf = CONFIG.copy()
    val_conf["context_csv_path"] = "data/validation.csv"
    val_dataset = FemurImageDataset(config=val_conf, split="val")
    model = UpsampleUNet(1,1, CONFIG)
    print(model(torch.randn(1,1,64,64,64)).shape)

    # Load model
    # model.load_state_dict(torch.load("models/BasicUNet_epoch_5.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    trainer = Trainer(model, dataset, val_dataset, torch.nn.L1Loss(), optimizer, CONFIG, test_data=val_dataset)
    trainer.train_test()
    
        
   



if __name__ == "__main__":
    main()
