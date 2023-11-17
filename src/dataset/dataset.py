import torch
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from skimage.util import random_noise
from .transforms import get_image_augmentation

class FemurImageDataset(Dataset):
    """Femur Image Dataset"""

    def __init__(self, config, split) -> None:
        super().__init__()
        assert type(config) == dict, "Config must be a dictionary."
        assert "context csv path" in config, "Config must contain a context csv path."
        assert split in ["train", "val", "test"], "Split must be one of [train, val, test]."
        self._config = config
        self._split = split

        if "base path" in config:
            base_path = config["base path"]

        if "output size" in config:
            self.output_size = config["output size"]
        else:
            self.output_size = [64, 64, 64] # Default output size in form [z, x, y]
        
        if "use accelerator" in config:
            self.use_accelerator = config["use accelerator"]
        else:
            self.use_accelerator = True

        if self.use_accelerator:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == torch.device("cpu"):
                self.device = torch.device("mps" if torch.backends.mps.available() else "cpu")
        else:
            self.device = torch.device("cpu")

        if "augmentation" in config and config["augmentation"] is not None:
            self.augmentation = get_image_augmentation(config, split)
        else:
            self.augmentation = get_image_augmentation(config, "test")
        
        if "add channel dim" in config:
            self.add_channel_dim = config["add channel dim"]
        
        self.sample_paths = pd.read_csv(config["context csv path"])
        self.PCCT_paths = self.sample_paths["PCCT_path"]
        self.HRpQCT_paths = self.sample_paths["HRpQCT_path"]
        if base_path is not None:
            self.PCCT_paths = base_path + self.PCCT_paths
            self.HRpQCT_paths = base_path + self.HRpQCT_paths

        self.length = 0
        self.slice_from_range = {}
        for i, (HRpQCT_folder, PCCT_folder) in enumerate(zip(self.HRpQCT_paths, self.PCCT_paths)):
            if not os.path.exists(HRpQCT_folder):
                raise ValueError(f"Folder {HRpQCT_folder} does not exist.")
            files_in_folder = len([file for file in os.listdir(HRpQCT_folder) if file.endswith(".npz")])
            pcct_files = len([file for file in os.listdir(PCCT_folder) if file.endswith(".npz")])
            
            assert files_in_folder == pcct_files, f"Number of files in {HRpQCT_folder} and {PCCT_folder} do not match."

            self.slice_from_range[HRpQCT_folder] = (self.length, self.length + files_in_folder, i)
            self.length += files_in_folder


            
    
    def __len__(self):
        return self.length
    
    def _load_range_from_folder(self, PCCT_folder, HRpQCT_folder, start, end):

        PCCT_images = []
        PCCT_folder = [os.path.join(PCCT_folder, file) for file in os.listdir(PCCT_folder)]
        HRpQCT_images = []
        HRpQCT_folder = [os.path.join(HRpQCT_folder, file) for file in os.listdir(HRpQCT_folder)]
        
        assert start >= 0, "Start index must be greater than 0."
        
        # Calculate padding needed to get to the expected size
        to_extend_back = 0
        if end > len(PCCT_folder):
            to_extend_back = end - len(PCCT_folder)
            end = len(PCCT_folder)

        # TODO: Multithreading? Time this.
        for i in range(start, end):
            PCCT_images.append(np.load(PCCT_folder[i])["arr_0"])
            HRpQCT_images.append(np.load(HRpQCT_folder[i])["arr_0"])

        # PCCT_images = [np.zeros_like(PCCT_images[0]) for _ in range(to_extend_start)] + PCCT_images
        PCCT_images = PCCT_images + [np.zeros_like(PCCT_images[0]) for _ in range(to_extend_back)]
        HRpQCT_images =  HRpQCT_images + [np.zeros_like(HRpQCT_images[0]) for _ in range(to_extend_back)]

        return PCCT_images, HRpQCT_images
    
    
    def __getitem__(self, index):
        folder = None
        sample = None
        for name, (start, end, folder_index) in self.slice_from_range.items():
            if start <= index < end:
                folder = name
                sample = folder_index
                break
        
        if folder is None:
            raise ValueError(f"Index {index} is out of range.")
        
        # Set index to be relative to the folder (i.e. refer to the first slice to be loaded from the folder)
        index = index - self.slice_from_range[folder][0]
        # Get the folder paths for the PCCT and HRpQCT images
        PCCT_path = self.PCCT_paths[sample]
        HRpQCT_path = folder
        
        # Load the images
        PCCT_images, HRpQCT_images = self._load_range_from_folder(PCCT_path, HRpQCT_path, index, index + self.output_size[0])
        PCCT_images = np.stack(PCCT_images, axis=0)
        HRpQCT_images = np.stack(HRpQCT_images, axis=0)
        # Apply augmentation

        PCCT_images = np.expand_dims(PCCT_images,0)
        HRpQCT_images = np.expand_dims(HRpQCT_images, 0)
        if self.augmentation is not None:
            aug = self.augmentation({"image": PCCT_images, "labels": HRpQCT_images})
            PCCT_images = aug["image"]
            HRpQCT_images = aug["labels"]

        # If images are not tensors convert them to tensors
        if not torch.is_tensor(PCCT_images):
            PCCT_images = torch.from_numpy(PCCT_images)
        if not torch.is_tensor(HRpQCT_images):
            HRpQCT_images = torch.from_numpy(HRpQCT_images)

        PCCT_images = PCCT_images.to(self.device, dtype=torch.float32)
        HRpQCT_images = HRpQCT_images.to(self.device, dtype=torch.float32)
        return PCCT_images, HRpQCT_images



        