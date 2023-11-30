import torch
import SimpleITK as sitk
import numpy as np
import os
import zipfile
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
import time

from skimage.util import random_noise
from .transforms import get_image_augmentation
from .utils import get_inital_crop_size

class FemurImageDataset(Dataset):
    """Femur Image Dataset"""

    def __init__(self, config, split) -> None:
        super().__init__()
        assert type(config) == dict, "Config must be a dictionary."
        assert "context_csv_path" in config, "Config must contain a context csv path."
        assert split in ["train", "val", "test"], "Split must be one of [train, val, test]."

        self._config = config
        self._split = split
        self._use_accelerator = config["use_accelerator"]
        self._input_size = [get_inital_crop_size(config["input_size"]) for i in range(3)]
        self._output_size = [get_inital_crop_size(config["output_size"]) for i in range(3)]

        # 
        self._scale_factor = [a/b for a,b in zip(self._output_size, self._input_size)]

        self._base_path = config["base_path"]
        
        if self._use_accelerator:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self._device == torch.device("cpu"):
                self._device = torch.device("mps" if torch.backends.mps.available() else "cpu")
        else:
            self._device = torch.device("cpu")

        if config["augmentation"]:
            self.augmentation = get_image_augmentation(config, split)
        else:
            self.augmentation = get_image_augmentation(config, "test")
        
        self.sample_paths = pd.read_csv(config["context_csv_path"])
        self.PCCT_paths = self.sample_paths["PCCT_path"]
        self.HRpQCT_paths = self.sample_paths["HRpQCT_path"]
        if self._base_path is not None:
            self.PCCT_paths = self._base_path + self.PCCT_paths
            self.HRpQCT_paths = self._base_path + self.HRpQCT_paths

        self.length = 0
        self.slice_from_range = {}
        for i in range(len(self.PCCT_paths)):
            PCCT_folder = self.PCCT_paths[i]
            HRpQCT_folder = self.HRpQCT_paths[i]
            if not os.path.exists(HRpQCT_folder):
                raise ValueError(f"Folder {HRpQCT_folder} does not exist.")
            
            # Get Sample name
            name = os.path.basename(HRpQCT_folder)
            
            # get zip files
            HRpQCT_zip = os.path.join(HRpQCT_folder, f"{name}.zip")
            PCCT_zip = os.path.join(PCCT_folder, f"{name}.zip")

            # Check if zip files exist
            if not os.path.exists(HRpQCT_zip):
                raise ValueError(f"Zip file {HRpQCT_zip} does not exist.")
            if not os.path.exists(PCCT_zip):
                raise ValueError(f"Zip file {PCCT_zip} does not exist.")
            
            # Get number of files in zip files
            with zipfile.ZipFile(HRpQCT_zip, "r") as f:
                files_in_folder = len(f.infolist())
            with zipfile.ZipFile(PCCT_zip, "r") as f:
                pcct_files = len(f.infolist())
            print(f"Found {files_in_folder} files in {HRpQCT_folder} and {pcct_files} files in {PCCT_folder}")
            self.PCCT_paths[i] = os.path.join(PCCT_folder, f"{name}.zip")
            self.HRpQCT_paths[i] = os.path.join(HRpQCT_folder, f"{name}.zip")

            self.slice_from_range[self.PCCT_paths[i]] = (self.length, self.length + pcct_files, i)
            self.length += files_in_folder


            
    
    def __len__(self):
        return self.length
    
    def _load_range_from_folder(self, PCCT_folder, HRpQCT_folder, start, end):
        
        PCCT_images = []
        HRpQCT_images = []
        num_slices_in_folder = self.slice_from_range[PCCT_folder][1] - self.slice_from_range[PCCT_folder][0]
        
        assert start >= 0, "Start index must be greater than 0."
        # Calculate padding needed to get to the expected size
        to_extend_back = 0
        if end > num_slices_in_folder:
            to_extend_back = end - num_slices_in_folder
            end = num_slices_in_folder

        

        # TODO: Multithreading? Time this.
        with zipfile.ZipFile(PCCT_folder, "r") as f:
            for i in range(start, end):
                with f.open(f"{i}.npy") as file:
                    PCCT_images.append(np.load(file))


        # Load double the range from the HRpQCT folder
        with zipfile.ZipFile(HRpQCT_folder, "r") as f:
            for i in range(int(start*self._scale_factor[0]), int(end*self._scale_factor[0])+1):
                with f.open(f"{i}.npy") as file:
                    HRpQCT_images.append(np.load(file))
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
        PCCT_path = folder
        HRpQCT_path = self.HRpQCT_paths[sample]
        # Load the images
        PCCT_images, HRpQCT_images = self._load_range_from_folder(PCCT_path, HRpQCT_path, index, index + self._input_size[0])
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

        PCCT_images = PCCT_images.to(self._device, dtype=torch.float32)
        HRpQCT_images = HRpQCT_images.to(self._device, dtype=torch.float32)
        return PCCT_images, HRpQCT_images



        