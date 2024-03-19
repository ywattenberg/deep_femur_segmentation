import numpy as np
import SimpleITK as sitk
from skimage.metrics import normalized_mutual_information, structural_similarity
from functools import partial


def registration_score(fixed_image: sitk.Image | str , moving_image: sitk.Image | str, score_function: callable = None) -> float:
    """
    Calculate the registration score of two images.
    :param fixed_image: The fixed image.
    :param moving_image: The moving image.
    :return: The registration score.
    """
    if score_function is None:
        score_function = partial(structural_similarity, win_size=5)

    if isinstance(fixed_image, str):
        fixed_image = sitk.ReadImage(fixed_image)
    if isinstance(moving_image, str):
        moving_image = sitk.ReadImage(moving_image)
    fixed_image = sitk.GetArrayFromImage(fixed_image)
    moving_image = sitk.GetArrayFromImage(moving_image)
    
    return score_function(fixed_image, moving_image)


