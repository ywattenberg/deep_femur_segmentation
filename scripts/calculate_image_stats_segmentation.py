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
    # config["context_csv_path"] = r"HRpQCT_aim\\Cropped_regions.csv"
    if "seed" in config:
        torch.manual_seed(config["seed"])
    dataset = FemurSegmentationDataset(config, split="test")
    for i in range(len(dataset)):
        input, target, mask = dataset[i]
        pcct = input.squeeze().numpy().astype(float)
        hr = target.squeeze().numpy().astype(float)
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(pcct[36], cmap="gray")
        axs[1].imshow(hr[75], cmap="gray")
        plt.show()
        print(f"PCCT min/max: {pcct.min()} - {pcct.max()}")
        print(f"PCCT mean/std: {pcct.mean()} - {pcct.std()}")
        print(f"HR min/max: {hr.min()} - {hr.max()}")
        print(f"HR mean/std: {hr.mean()} - {hr.std()}")


    

if __name__ == "__main__":
    main()




    
