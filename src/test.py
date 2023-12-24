from typing import Any
import torch
import yaml
import monai.networks.nets as monai_nets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from trainer import Trainer
from dataset.dataset import FemurImageDataset 
from dataset.transforms import CustomCropRandomd
from model.basic_UNet import BasicUNet, UpsampleUNet



def main():
    torch.set_default_dtype(torch.float16)
    # Get Unet model from Monai
    CONFIG = yaml.safe_load(open("config/config.yaml", "r"))
    if "seed" in CONFIG:
        torch.manual_seed(CONFIG["seed"])
    print(CONFIG)
    dataset = FemurImageDataset(config=CONFIG, split="train")
    val_conf = CONFIG.copy()
    val_conf["context_csv_path"] = "data/validation.csv"
    val_dataset = FemurImageDataset(config=val_conf, split="val")
    model = UpsampleUNet(1,1, CONFIG)
    # Load model
    # model.load_state_dict(torch.load("models/BasicUNet_epoch_5.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    trainer = Trainer(model, dataset, val_dataset, torch.nn.L1Loss(), optimizer, CONFIG)
    trainer.train_test()
    
        
   



if __name__ == "__main__":
    main()
