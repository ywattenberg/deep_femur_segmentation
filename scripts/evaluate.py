import SimpleITK as sitk
import numpy as np
import os 
import torch
import sys
import argparse
import pandas as pd
from monai.metrics import DiceMetric
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import MeanIoU


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dataset.dataset_segmentation import FemurSegmentationDataset

def main(config, mask_dir):
    """
    Calculate the Dice similarity coefficient and Jacard similarity coefficient for trabecular and cortical masks
    """
    # Create Dataset 
    dataset = FemurSegmentationDataset(config, split="test")
    dir = os.listdir(mask_dir)
    df = pd.DataFrame(columns=["Trab Dice", "Cort Dice", "Trab Jaccard", "Cort Jaccard", "Trab Hausdorff", "Cort Hausdorff"])
    for i in range(len(dataset)):
        cort_mask, trab_mask = sorted([mask for mask in dir if mask.startswith(f"pred_mask_{i}")])
        cort_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, cort_mask)))
        trab_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, trab_mask)))

        _, _, mask = dataset[i]
        mask = mask.numpy()
        true_cort_mask = mask[0]
        true_trab_mask = mask[1]

        # Calculate Dice similarity coefficient
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        cort_dice = dice_metric(torch.tensor(cort_mask), torch.tensor(true_cort_mask))
        trab_dice = dice_metric(torch.tensor(trab_mask), torch.tensor(true_trab_mask))
        print(f"Trabecular Dice Similarity Coefficient: {trab_dice}, Cortical Dice Similarity Coefficient: {cort_dice}")

        # Calculate Jaccard similarity coefficient
        jaccard_metric = MeanIoU(include_background=True, reduction="mean")
        cort_jaccard = jaccard_metric(torch.tensor(cort_mask), torch.tensor(true_cort_mask))
        trab_jaccard = jaccard_metric(torch.tensor(trab_mask), torch.tensor(true_trab_mask))

        print(f"Trabecular Jaccard Similarity Coefficient: {trab_jaccard}, Cortical Jaccard Similarity Coefficient: {cort_jaccard}")

        # Calculate Hausdorff distance
        hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")
        cort_hausdorff = hausdorff_metric(torch.tensor(cort_mask), torch.tensor(true_cort_mask))
        trab_hausdorff = hausdorff_metric(torch.tensor(trab_mask), torch.tensor(true_trab_mask))

        print(f"Trabecular Hausdorff Distance: {trab_hausdorff}, Cortical Hausdorff Distance: {cort_hausdorff}")

        df = df.append({"Trab Dice": trab_dice, "Cort Dice": cort_dice, "Trab Jaccard": trab_jaccard, "Cort Jaccard": cort_jaccard, "Trab Hausdorff": trab_hausdorff, "Cort Hausdorff": cort_hausdorff}, ignore_index=True)
    
    df.to_csv(os.path.join(mask_dir, "metrics.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="Path to the config file")
    ap.add_argument("-m", "--mask_dir", required=True, help="Path to the directory containing the masks")
    args = ap.parse_args()
    
    main(args.config, args.mask_dir)

    






        





