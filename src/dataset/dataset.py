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

    def __init__(self, csv_path, base_path=None, image_transform=None, PCCT_transform=None,  use_accelerator=True, number_of_context_slices=5, add_channel_dim=True) -> None:
        super().__init__()
        self.context_csv = csv_path
        self.PCCT_transform = PCCT_transform
        self.number_of_context_slices = number_of_context_slices
        self.add_channel_dim = add_channel_dim

        if image_transform is None:
            self.image_transform = get_image_transform()
        else:
            self.image_transform = image_transform

        if use_accelerator:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == torch.device("cpu"):
                self.device = torch.device("mps" if torch.backends.mps.available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        self.sample_paths = pd.read_csv(csv_path)
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


            
    
    def __len__(self):
        return self.length
    
    def _load_range_from_folder(self, PCCT_folder, HRpQCT_folder, start, end):

        PCCT_images = []
        PCCT_folder = [os.path.join(PCCT_folder, file) for file in os.listdir(PCCT_folder)]

        # Make sure that the range is valid

        HRpQCT_folder = [os.path.join(HRpQCT_folder, file) for file in os.listdir(HRpQCT_folder)]
        HRpQCT_image = np.load(HRpQCT_folder[start + self.number_of_context_slices])["arr_0"]
        

        to_extend_start, to_extend_back = (0,0)
        if start < 0:
            to_extend_start = np.abs(start)
            start = 0
            
        if end > len(PCCT_folder):
            to_extend_back = end - len(PCCT_folder)
            end = len(PCCT_folder)
        

        for i in range(start, end):
            PCCT_images.append(np.load(PCCT_folder[i])["arr_0"])
        PCCT_images = [np.zeros_like(PCCT_images[0]) for _ in range(to_extend_start)] + PCCT_images
        PCCT_images = PCCT_images + [np.zeros_like(PCCT_images[0]) for _ in range(to_extend_back)]


        return PCCT_images, HRpQCT_image
    
    
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
        PCCT_images, HRpQCT_image = self._load_range_from_folder(PCCT_path, HRpQCT_path, index - self.number_of_context_slices, index + self.number_of_context_slices+1)

        PCCT_images = np.stack(PCCT_images, axis=0)
        PCCT_images = torch.from_numpy(PCCT_images).float()

        HRpQCT_image = torch.from_numpy(HRpQCT_image).float()
        if self.image_transform is not None:
            # Stack images to apply the same transform to all images
            images = torch.cat([PCCT_images, HRpQCT_image.unsqueeze(0)], dim=0)
            images = self.image_transform(images)
            # Split images again
            PCCT_images = images[:-1]
            HRpQCT_image = images[-1]

        if self.PCCT_transform is not None:
            PCCT_images = self.PCCT_transform(PCCT_images)

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


        