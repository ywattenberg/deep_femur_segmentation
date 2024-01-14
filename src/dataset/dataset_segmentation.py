import torch
import pandas as pd
import numpy as np
import yaml
import os
from .transforms import get_image_segmentation_augmentation

class FemurSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, config, split="train"):
        assert split in ["train", "val", "test"], f"Split {split} not supported"
        self.config = config
        self.split = split
        self.with_cort_and_trab = config["use_cortical_and_trabecular"]
        self.base_path = config["base_path"]
        context_csv_path = os.path.join(self.base_path, config["context_csv_path"])
        self.context = pd.read_csv(context_csv_path, delimiter=";")

        self.data = [folder for folder in os.listdir(self.base_path) if not self.with_cort_and_trab or not folder.endswith("h")]
        print(self.data)
        valid_samples = set([i.lower() for i in list(self.context["SampName"].values)])
        self.data = [folder for folder in self.data if folder.lower()[:-3] in valid_samples]
         
        if config["augmentation"]:
            self.augmentation = get_image_segmentation_augmentation(config, split)
        else:
            self.augmentation = get_image_segmentation_augmentation(config, "test")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        folder = self.data[index]
        full_path = os.path.join(self.base_path, folder)
        with open(os.path.join(full_path, "image_stats.yaml"), "r") as f:
            image_stats = yaml.load(f, Loader=yaml.FullLoader)
        sample_name = image_stats["name"]
        image = np.load(os.path.join(full_path, f"{sample_name}_image.npy"))
        mask = np.load(os.path.join(full_path, f"{sample_name}_outer_dicom.npy"))

        mask_offset = image_stats["outer_offset"]
        offset_mask = np.zeros_like(image, dtype=bool)
        offset_mask[:, mask_offset[0]:mask_offset[0]+mask.shape[1], mask_offset[1]:mask_offset[1]+mask.shape[2]] = mask
        offset_mask = np.expand_dims(offset_mask, 0)

        aug_dict = {"image": image, "mask": offset_mask}

        if self.with_cort_and_trab:

            cortical = np.load(os.path.join(full_path, f"{sample_name}_cortical.npy"))
            trabecular = np.load(os.path.join(full_path, f"{sample_name}_trabecular.npy"))

            cortical_offset = image_stats["cortical_offset"]
            trabecular_offset = image_stats["trabecular_offset"]
            offset_cortical = np.zeros_like(image, dtype=bool)
            offset_trabecular = np.zeros_like(image, dtype=bool)

            offset_cortical[:, cortical_offset[0]:cortical_offset[0]+cortical.shape[1], cortical_offset[1]:cortical_offset[1]+cortical.shape[2]] = cortical
            offset_trabecular[:, trabecular_offset[0]:trabecular_offset[0]+trabecular.shape[1], trabecular_offset[1]:trabecular_offset[1]+trabecular.shape[2]] = trabecular
            offset_cortical = np.expand_dims(offset_cortical, 0)
            offset_trabecular = np.expand_dims(offset_trabecular, 0)

            aug_dict["cortical"] = offset_mask * offset_cortical
            aug_dict["trabecular"] = offset_mask * offset_trabecular

        aug_dict["image"] = np.expand_dims(aug_dict["image"], 0)
        
        # Apply transforms
        while True:
            cropped_dict = self.augmentation[1](aug_dict)
            if cropped_dict["mask"].sum() > np.prod(self.config["input_size"]) * 0.1:
                break
    
        aug_dict = self.augmentation[0](cropped_dict)
        out = list(aug_dict.values())
        image = out[0]
        mask = torch.cat(out[1:], dim=0)
        return image, mask

        


        

    


        

