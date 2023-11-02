import torch
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

class FemurImageDataset(Dataset):
    """Femur Image Dataset"""

    def __init__(self, csv_path, base_path=None, image_transform=None, label_transform=None,  use_gpu=True, number_of_context_slices=5) -> None:
        super().__init__()
        self.context_csv = csv_path
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.number_of_context_slices = number_of_context_slices
        self.sample_paths = pd.read_csv(csv_path)
        self.sample_paths = self.sample_paths.to_numpy()
        self.PCCT_paths = self.sample_paths[:,0]
        self.HRpQCT_paths = self.sample_paths[:,1]
        if base_path is not None:
            self.PCCT_paths = [os.path.join(base_path, path) for path in self.PCCT_paths]
            self.HRpQCT_paths = [os.path.join(base_path, path) for path in self.HRpQCT_paths]
        self.length = 0
        self.sample_from_range = {}
        for i, folder in enumerate(self.HRpQCT_paths):
            if not os.path.exists(folder):
                raise ValueError(f"Folder {folder} does not exist.")
            files_in_folder = len([file for file in os.listdir(folder) if file.endswith(".npz")])
            self.sample_from_range[folder] = (self.length, self.length + files_in_folder, i)
            self.length += files_in_folder
    
    def __len__(self):
        return self.length
    
    def _load_range_from_folder(self, PCCT_folder, HRpQCT_folder, start, end):
        PCCT_images = []
        HRpQCT_images = []
        PCCT_folder = [os.path.join(PCCT_folder, file) for file in os.listdir(PCCT_folder)]
        HRpQCT_folder = [os.path.join(HRpQCT_folder, file) for file in os.listdir(HRpQCT_folder)]

        for i in range(start, end):
            PCCT_folder.append(np.load(PCCT_folder[i]))
            HRpQCT_folder.append(np.load(HRpQCT_folder[i]))

        return PCCT_images, HRpQCT_images

    
    def __getitem__(self, index):
        folder = None
        sample = None
        for name, (start, end, index) in self.sample_from_range.items():
            if start <= index < end:
                folder = name
                sample = index
                break
        
        if folder is None:
            raise ValueError(f"Index {index} is out of range.")
        
        index = index - self.sample_from_range[folder][0]
        PCCT_path = self.PCCT_paths[sample]
        HRpQCT_path = folder
        PCCT_images, HRpQCT_images = self._load_range_from_folder(PCCT_path, HRpQCT_path, index, index + self.number_of_context_slices)
        PCCT_images = np.stack(PCCT_images, axis=0)
        HRpQCT_images = np.stack(HRpQCT_images, axis=0)
        PCCT_images = torch.from_numpy(PCCT_images).float()
        HRpQCT_images = torch.from_numpy(HRpQCT_images).float()

        if self.image_transform is not None:
            PCCT_images = self.image_transform(PCCT_images)
            HRpQCT_images = self.image_transform(HRpQCT_images)

        return PCCT_images, HRpQCT_images


        