import numpy as np
import SimpleITK as sitk
from sklearn.metrics import normalized_mutual_info_score


def registration_score(fixed_image: sitk.Image | str , moving_image: sitk.Image | str):
    """
    Calculate the registration score of two images.
    :param fixed_image: The fixed image.
    :param moving_image: The moving image.
    :return: The registration score.
    """
    if isinstance(fixed_image, str):
        fixed_image = sitk.ReadImage(fixed_image)
    if isinstance(moving_image, str):
        moving_image = sitk.ReadImage(moving_image)
    fixed_image = sitk.GetArrayFromImage(fixed_image)
    moving_image = sitk.GetArrayFromImage(moving_image)
    
    return normalized_mutual_info_score(fixed_image, moving_image)




