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
        image = np.load(os.path.join(full_path, f"{sample_name}_seg.npy"))
        mask = np.load(os.path.join(full_path, f"{sample_name}_outer.npy"))

        aug_dict = {"image": image, "mask": mask}

        if self.with_cort_and_trab:
            cortical = np.load(os.path.join(full_path, f"{sample_name}_cortical.npy"))
            trabecular = np.load(os.path.join(full_path, f"{sample_name}_trabecular.npy"))
            aug_dict["cortical"] = cortical
            aug_dict["trabecular"] = trabecular

        min_size = list(image.shape)
        for key in aug_dict.keys():
            for i in range(3):
                min_size[i] = min(min_size[i], aug_dict[key].shape[i])

        # Crop image and mask to same size
        for key in aug_dict.keys():
            aug_dict[key] = aug_dict[key][0:min_size[0], 0:min_size[1], 0:min_size[2]]

        aug_dict["image"] = np.expand_dims(aug_dict["image"], 0)
        
        # Apply transforms
        aug_dict = self.augmentation(aug_dict)
        out = list(aug_dict.values())
        image = out[0]
        mask = torch.stack(out[1:])
        return image, mask

        


        

    


        

