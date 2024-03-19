import torch
import yaml
import monai.networks.nets as monai_nets

from monai.losses import DiceLoss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.trainer_2D_seg import Trainer
from src.dataset.dataset_segmentation import FemurSegmentationDataset 
from src.loss.DiceL1Loss import DiceL1Loss
from src.model.retina_UNet import Retina_UNet

import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float32)


def main(config_path, epochs_between_test, base_path):
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
    config = yaml.safe_load(open(config_path, "r"))
    config["base_path"] = base_path

    # config["context_csv_path"] = r"HRpQCT_aim\\Cropped_regions.csv"
    if "seed" in config:
        torch.manual_seed(config["seed"])
    dataset = FemurSegmentationDataset(config, split="train")

    val_config = config.copy()
    val_config["context_csv_path"] = r"numpy/Cropped_regions_val.csv"
    val_dataset = FemurSegmentationDataset(val_config, split="train")
    # model = Retina_UNet(
    #     in_channels=1,
    #     out_channels_mask=2 if config["use_cortical_and_trabecular"] else 1,
    #     out_channels_upsample=1,
    #     config=config,
    # )
    model = monai_nets.UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=2 if config["use_cortical_and_trabecular"] else 1,
        channels=config["model"]["features"],
        strides=config["model"]["strides"],
        dropout=config["model"]["dropout"],
        norm=config["model"]["norm"],
        act=config["model"]["activation"],
    )
    model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    test_dataset = FemurSegmentationDataset(config, split="test")
    trainer = Trainer(model, dataset, val_dataset, loss_fn, optimizer, config)
    trainer.train_test(epochs_between_test=epochs_between_test)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, default="config/segmentation_config.yaml")
    argparser.add_argument("--tmp_dir", type=str)
    argparser.add_argument("--epochs_between_test", type=int, default=10)
    args = argparser.parse_args()

    main(args.config, args.epochs_between_test, args.tmp_dir)