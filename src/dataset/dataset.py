import torch
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from skimage.util import random_noise

class FemurImageDataset(Dataset):
    """Femur Image Dataset"""

    def __init__(self, config, split) -> None:
        super().__init__()
        assert type(config) == dict, "Config must be a dictionary."
        assert "context csv path" in config, "Config must contain a context csv path."
        assert split in ["train", "val", "test"], "Split must be one of [train, val, test]."
        self._config = config
        context_csv = pd.read_csv(config["context csv path"])

        if "base path" in config:
            base_path = config["base path"]

        if "output size" in config:
            self.output_size = config["output size"]
        else:
            self.output_size = [64, 64, 64] # Default output size in form [x, y, z]
        
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

        if "transform" in config:
            self.transform = get_image_transform(split)
        if "augmentation" in config:
            self.augmentation = get_image_augmentation(split)
        
        self.sample_paths = pd.read_csv(context_csv)
        self.PCCT_paths = self.sample_paths["PCCT_path"]
        self.HRpQCT_paths = self.sample_paths["HRpQCT_path"]
        if base_path is not None:
            self.PCCT_paths = base_path + self.PCCT_paths
            self.HRpQCT_paths = base_path + self.HRpQCT_paths

        self.length = 0
        self.sample_from_range = {}
        for i, (HRpQCT_folder, PCCT_folder) in enumerate(zip(self.HRpQCT_paths, self.PCCT_paths)):
            if not os.path.exists(HRpQCT_folder):
                raise ValueError(f"Folder {HRpQCT_folder} does not exist.")
            files_in_folder = len([file for file in os.listdir(HRpQCT_folder) if file.endswith(".npz")])
            pcct_files = len([file for file in os.listdir(PCCT_folder) if file.endswith(".npz")])
            
            assert files_in_folder == pcct_files, f"Number of files in {HRpQCT_folder} and {PCCT_folder} do not match."

            self.sample_from_range[HRpQCT_folder] = (self.length, self.length + files_in_folder, i)
            self.length += files_in_folder
        self.length -= 2 * self.output_size[-1]


            
    
    def __len__(self):
        return self.length
    
    def _load_range_from_folder(self, PCCT_folder, HRpQCT_folder, start, end):

        PCCT_images = []
        PCCT_folder = [os.path.join(PCCT_folder, file) for file in os.listdir(PCCT_folder)]

        # Make sure that the range is valid

        HRpQCT_folder = [os.path.join(HRpQCT_folder, file) for file in os.listdir(HRpQCT_folder)]
        HRpQCT_images = []

        to_extend_start, to_extend_back = (0,0)
        if start < 0:
            to_extend_start = np.abs(start)
            start = 0
            
        if end > len(PCCT_folder):
            to_extend_back = end - len(PCCT_folder)
            end = len(PCCT_folder)
        assert to_extend_back == 0, "Not implemented yet."
        assert to_extend_start == 0, "Not implemented yet."

        # TODO: Multithreading? Time this.
        for i in range(start, end):
            PCCT_images.append(np.load(PCCT_folder[i])["arr_0"])
            HRpQCT_images.append(np.load(HRpQCT_folder[i])["arr_0"])
        # PCCT_images = [np.zeros_like(PCCT_images[0]) for _ in range(to_extend_start)] + PCCT_images
        # PCCT_images = PCCT_images + [np.zeros_like(PCCT_images[0]) for _ in range(to_extend_back)]

        return PCCT_images, HRpQCT_images
    
    
    def __getitem__(self, index):
        folder = None
        sample = None
        for name, (start, end, folder_index) in self.sample_from_range.items():
            if start <= index < end:
                folder = name
                sample = folder_index
                break
        
        if folder is None:
            raise ValueError(f"Index {index} is out of range.")
        
        index = index - self.sample_from_range[folder][0]

        PCCT_path = self.PCCT_paths[sample]
        HRpQCT_path = folder
        PCCT_images, HRpQCT_images = self._load_range_from_folder(PCCT_path, HRpQCT_path, index - self.output_size[-1], index + self.output_size[-1])

        PCCT_images = np.stack(PCCT_images, axis=0)
        HRpQCT_images = np.stack(HRpQCT_images, axis=0)

        # Apply transform
        if self.transform is not None:
            PCCT_images = self.transform(PCCT_images, HRpQCT_images)

        # Apply augmentation
        if self.augmentation is not None:
            PCCT_images = self.augmentation(PCCT_images)
            # HRpQCT_images = self.augmentation(HRpQCT_images) 

        if self.add_channel_dim:
            PCCT_images = PCCT_images.unsqueeze(0)
            HRpQCT_image = HRpQCT_image.unsqueeze(0)

        PCCT_images = PCCT_images.to(self.device)
        HRpQCT_image = HRpQCT_image.to(self.device)
        return PCCT_images, HRpQCT_image.unsqueeze(0)

def get_image_transform():
    return v2.Compose([
        v2.RandomRotation(90),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
    ])


        