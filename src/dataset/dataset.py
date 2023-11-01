import torch
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset

class FemurImageDataset(Dataset):
    """Femur Image Dataset"""

    def __init__(self, csv_path, image_transform=None, label_transform=None,  use_gpu=True, number_of_context_slices=5) -> None:
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
        self.length = 0
        for folder in self.HRpQCT_paths:
            if not os.path.isdir(folder):
                raise ValueError(f"Folder {folder} does not exist.")
            self.length += len([file for file in os.listdir(folder) if file.endswith(".npz")])


        