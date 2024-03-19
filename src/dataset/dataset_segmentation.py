import torch
import pandas as pd
import numpy as np
import yaml
import os


from .transforms import get_image_segmentation_augmentation
from .utils import load_slices_region, get_padded_mask, maximum_overlapping_rectangle, load_slices, fix_mask_depth

class FemurSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, config, split="train"):
        assert split in ["train", "val", "test"], f"Split {split} not supported"

        # Load config
        self.config = config
        self.split = split
        self.with_cort_and_trab = config["use_cortical_and_trabecular"]
        self.base_path = config["base_path"]
        self.mask_path = os.path.join(self.base_path, config["mask_path"])
        
        context_csv_path = os.path.join(self.base_path, config["context_csv_path"])
        self.context = pd.read_csv(context_csv_path, delimiter=";")

        # List all samples and filter if necessary (we don't have cortical and trabecular masks for regions at femur head)
        self.data = [f for f in os.listdir(self.mask_path) if os.path.isdir(os.path.join(self.mask_path, f))]
        self.data = [f for f in self.data if not self.with_cort_and_trab or not f.endswith("h")]
        # Further assert that all samples are in the context csv and thus have a PCCT region defined
        # This is basically the train val test split
        valid_samples = set([i.lower() for i in list(self.context["SampName"].values)])

        self.data = [f for f in self.data if f.lower()[:-3] in valid_samples]

        if config["augmentation"]:
            self.augmentation = get_image_segmentation_augmentation(config, split)
        else:
            self.augmentation = get_image_segmentation_augmentation(config, "test")


    def __len__(self):
        return len(self.data)
    

    def _load_sample(self, index):
        """ Loads the given sample and masks from the dataset without any augmentation. """
        full_path = os.path.join(self.base_path, self.mask_path, self.data[index])
        with open(os.path.join(full_path, "image_stats.yaml"), "r") as f:
            image_stats = yaml.load(f, Loader=yaml.FullLoader)
        sample_name = image_stats["name"]
    
        # Load HR-pQCT image and mask
        image = np.load(os.path.join(full_path, f"{sample_name}_image.npy"))
        image = np.expand_dims(image, 0)
        pcct = np.load(os.path.join(full_path, f"{sample_name}_pcct.npy"))
        pcct = np.expand_dims(pcct, 0)
        mask = np.load(os.path.join(full_path, f"{sample_name}_threshold.npy"))
        mask = np.expand_dims(mask, 0)
        mask = torch.tensor(mask,  dtype=torch.uint8)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0)
        
        sample = {"image": image, "mask": mask, "pcct": pcct}

        if self.with_cort_and_trab:
            cortical = np.load(os.path.join(full_path, f"{sample_name}_cortical.npy"))
            if cortical.shape[0] < image.shape[1]:
                print(f"Corrupted for cort {sample_name} at {index}")
            cortical = fix_mask_depth(cortical, image.shape[1:])
            trabecular = np.load(os.path.join(full_path, f"{sample_name}_trabecular.npy"))
            if trabecular.shape[0] < image.shape[1]:
                print(f"Corrupted for trab {sample_name} at {index}")
            trabecular = fix_mask_depth(trabecular, image.shape[1:])

            cortical_offset = image_stats["cortical_offset"]
            trabecular_offset = image_stats["trabecular_offset"]

            offset_cortical = get_padded_mask(cortical, image.shape, cortical_offset)
            offset_trabecular = get_padded_mask(trabecular, image.shape, trabecular_offset)

            offset_cortical = torch.tensor(offset_cortical, dtype=torch.uint8)
            offset_trabecular = torch.tensor(offset_trabecular,  dtype=torch.uint8 )
            offset_cortical = torch.nn.functional.interpolate(offset_cortical.unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0)
            offset_trabecular = torch.nn.functional.interpolate(offset_trabecular.unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze(0)


            sample["cortical"] = offset_cortical
            sample["trabecular"] = offset_trabecular 
        
        return sample


    def __getitem__(self, index):
        """ Loads the given sample and masks from the dataset. """
        aug_dict = self._load_sample(index)
        if aug_dict is None:
            return None, None, None
        
        # print([(k, v.shape) for k,v in cropped_dict.items()])
        print(f"Augmenting {index}")
        aug_dict = self.augmentation(aug_dict)
        print(f"Done Augmented {index}")
        image = aug_dict["image"]
        pcct = aug_dict["pcct"]
        masks = [aug_dict["mask"]]
        if self.with_cort_and_trab:
            masks = []
            masks.append(aug_dict["cortical"] )
            masks.append(aug_dict["trabecular"])
            masks.append(aug_dict["mask"])
        mask = torch.cat(masks, dim=0).to(torch.float32)
        # print(f"Mask {mask.shape}")
        # print(f"PCCT shape {pcct.shape}")
        # print(f"image shape {image.shape}")
        return pcct, image, mask

        

        

    


        

