import SimpleITK as sitk
import numpy as np
import os 
import torch
import sys
import argparse
import yaml
import pandas as pd
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU, SurfaceDiceMetric
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dataset.dataset_segmentation import FemurSegmentationDataset

def main(mask_dir):
    """
    Calculate the Dice similarity coefficient and Jacard similarity coefficient for trabecular and cortical masks
    """
    # Create Dataset 
    dir = os.listdir(mask_dir)
    # ["Trab Dice", "Cort Dice", "Trab Jaccard", "Cort Jaccard", "Trab Hausdorff", "Cort Hausdorff"]
    dict = {key: [] for key in ["Trab Dice", "Cort Dice", "Trab Jaccard", "Cort Jaccard", "Trab Hausdorff", "Cort Hausdorff", "Trab Surface Dice", "Cort Surface Dice"]}
    for i in range(4):
        true_cort_mask, true_trab_mask = sorted([mask for mask in dir if mask.startswith(f"mask_{i}")])
        cort_mask, trab_mask = sorted([mask for mask in dir if mask.startswith(f"pred_mask_{i}")])
        true_cort_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, true_cort_mask))).astype(np.uint8)
        true_trab_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, true_trab_mask))).astype(np.uint8)
        cort_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, cort_mask))).astype(np.uint8)
        trab_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir, trab_mask))).astype(np.uint8)

        true_cort_mask = torch.tensor(true_cort_mask)[None]
        true_trab_mask = torch.tensor(true_trab_mask)[None]

        cort_mask = torch.tensor(cort_mask)[None]
        trab_mask = torch.tensor(trab_mask)[None]


        # fig, ax = plt.subplots(2, 2)
        # ax[0][0].imshow(cort_mask[0], cmap="gray")
        # ax[0][0].set_title("Predicted Cortical Mask")
        # ax[0][1].imshow(trab_mask[0], cmap="gray")
        # ax[0][1].set_title("Predicted Trabecular Mask")
        # ax[1][0].imshow(true_cort_mask[0], cmap="gray")
        # ax[1][0].set_title("True Cortical Mask")
        # ax[1][1].imshow(true_trab_mask[0], cmap="gray")
        # ax[1][1].set_title("True Trabecular Mask")
        # plt.show()

        # Calculate Dice similarity coefficient
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        dice = dice_metric([cort_mask,trab_mask], [true_cort_mask, true_trab_mask])
        print(f"Cort, Trabecular Dice Similarity Coefficient: {dice}")
        
        # Calculate Jaccard similarity coefficient
        jaccard_metric = MeanIoU(include_background=False, reduction="mean")
        jaccard = jaccard_metric([cort_mask, trab_mask], [true_cort_mask, true_trab_mask])
        print(f"Cort, Trabecular Jaccard Similarity Coefficient: {jaccard}")
        

        # Calculate Hausdorff distance
        hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
        hausdorff = hausdorff_metric([cort_mask, trab_mask], [true_cort_mask, true_trab_mask])
        print(f"Cort, Trabecular Hausdorff Distance: {hausdorff}")

        # Surface Dice
        surface_dice = SurfaceDiceMetric(class_thresholds=[2], include_background=False, reduction="mean")
        surface_dice_metric = surface_dice([cort_mask, trab_mask], [true_cort_mask, true_trab_mask], include_background=False)
        print(f"Cort, Trabecular Surface Dice: {surface_dice_metric}")

        dict["Trab Dice"].append(np.round(dice[1].item(), decimals=3))
        dict["Cort Dice"].append(np.round(dice[0].item(), decimals=3))
        dict["Trab Jaccard"].append(np.round(jaccard[1].item(), decimals=3))
        dict["Cort Jaccard"].append(np.round(jaccard[0].item(), decimals=3))
        dict["Trab Hausdorff"].append(np.round(hausdorff[1].item(), decimals=3))
        dict["Cort Hausdorff"].append(np.round(hausdorff[0].item(), decimals=3))
        dict["Trab Surface Dice"].append(np.round(surface_dice_metric[1].item(), decimals=3))
        dict["Cort Surface Dice"].append(np.round(surface_dice_metric[0].item(), decimals=3))


    df = pd.DataFrame(dict)

    # Calculate mean and standard deviation
    mean = df.mean()
    std = df.std()
    print(f"Mean: {mean.round(3)}")
    print(f"Standard Deviation: {std.round(3)}")
    df.to_csv(os.path.join(mask_dir, "metrics.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mask_dir", required=True, help="Path to the directory containing the masks")
    args = ap.parse_args()
    
    main(args.mask_dir)

    






        





