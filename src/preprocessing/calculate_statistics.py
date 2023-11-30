import numpy as np
import SimpleITK as sitk
import os


from .utils import image_to_array, read_dicom_series

def calculate_statistics(image: sitk.Image | np.ndarray | str, isDicom: bool = False):
    """Calculate statistics of an image."""
    if isinstance(image, str):
        if isDicom:
            image = read_dicom_series(image)
        else:
            image = sitk.ReadImage(image)

    if isinstance(image, sitk.Image):
        image = image_to_array(image)
    
    return {
        "median": np.median(image),
        "mean": np.mean(image),
        "std": np.std(image),
        "min": np.min(image),
        "max": np.max(image),
        "percentile_99_5": np.percentile(image, 99.5),
        "percentile_00_5": np.percentile(image, 0.5),
        "total_volume": np.prod(image.shape),
    }

def calculate_statistics_from_folder(folder: str):
    parts = os.listdir(folder)
    parts = [part for part in parts if os.path.isdir(os.path.join(folder, part)) and part.startswith("part")]
    stats = []
    for part in parts:
        stats.append(calculate_statistics(os.path.join(folder, part)))
    
    # Combine statistics
    median = np.median([stat["median"] for stat in stats])
    mean = np.sum([stat["mean"] * stat["total_volume"] for stat in stats]) / np.sum([stat["total_volume"] for stat in stats])
    std = np.sqrt(np.sum([stat["std"]**2 * stat["total_volume"] for stat in stats]) / np.sum([stat["total_volume"] for stat in stats]))
    min = np.min([stat["min"] for stat in stats])
    max = np.max([stat["max"] for stat in stats])
    percentile_99_5 = np.percentile([stat["percentile_99_5"] for stat in stats], 99.5)
    percentile_00_5 = np.percentile([stat["percentile_00_5"] for stat in stats], 0.5)
    total_volume = np.sum([stat["total_volume"] for stat in stats])

    return {
        "median": median,
        "mean": mean,
        "std": std,
        "min": min,
        "max": max,
        "percentile_99_5": percentile_99_5,
        "percentile_00_5": percentile_00_5,
        "total_volume": total_volume,
    }
    
    
    