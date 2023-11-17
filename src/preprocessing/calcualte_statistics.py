import numpy as np
import SimpleITK as sitk
import os

from .utils import image_to_array

def _get_intensity_statistics(image: np.ndarray):
    return {
        "median": np.median(image),
        "mean": np.mean(image),
        "std": np.std(image),
        "min": np.min(image),
        "max": np.max(image),
        "percentile_99_5": np.percentile(image, 99.5),
        "percentile_00_5": np.percentile(image, 0.5)
    }

def calculate_statistics(image: sitk.Image | np.ndarray):
    if isinstance(image, sitk.Image):
        image = image_to_array(image)
    
    statistics = {
        "intensity": _get_intensity_statistics(image)
    }
    return statistics