import SimpleITK as sitk
import numpy as np
import argparse

from registration.registration_score import registration_score
from preprocessing.preprocessing import intensity_normalization


def main(fixed_image: sitk.Image | str , moving_image: sitk.Image | str, calculate_normalized_mutual_information: bool = False):
    """
    Calculate a very simple registration score of two images by moving the moving image in all directions by `voxel_size` and calculating the normalized mutual information score.
    :param fixed_image: The fixed image.
    :param moving_image: The moving image.
    :return: The registration score.
    """
    if isinstance(fixed_image, str):
        fixed_image = sitk.ReadImage(fixed_image)
    if isinstance(moving_image, str):
        moving_image = sitk.ReadImage(moving_image)

    if sitk.GetArrayFromImage(fixed_image).min() < 0:
        fixed_image = intensity_normalization(fixed_image)
        print("Fixed image was intensity normalized.")
    if sitk.GetArrayFromImage(moving_image).min() < 0:
        moving_image = intensity_normalization(moving_image)
        print("Moving image was intensity normalized.")
    
    if not calculate_normalized_mutual_information:
        score_function = lambda x, y: np.sum(np.abs(x - y))
    else:
        score_function = None
    base_score = registration_score(fixed_image, moving_image, score_function=score_function)
        
    print(f"Registration normalized mutual information score of original images: {base_score}")
    scores = []
    base_translation = [0.0, 0.0, 0.0]

    voxel_size = fixed_image.GetSpacing()
    for i in range(3):
        translation = base_translation.copy()
        translation[i] = voxel_size[i]
        translation_transform = sitk.TranslationTransform(3, translation)
        moving_image_transformed = sitk.Resample(moving_image, fixed_image, translation_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        scores.append(registration_score(fixed_image, moving_image_transformed, score_function=score_function))
        
        translation[i] = -voxel_size[i]
        translation_transform = sitk.TranslationTransform(3, translation)
        moving_image_transformed = sitk.Resample(moving_image, fixed_image, translation_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        scores.append(registration_score(fixed_image, moving_image_transformed, score_function=score_function))

    print(f"Registration normalized mutual information score of original images: {base_score}")
    print(f"Registration normalized mutual information score of images moved by one voxel in each direction min: {min(scores)}, max: {max(scores)}")
    print(f"Full array of scores: {scores}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate a very simple registration score of two images by moving the moving image in all directions by `voxel_size` and calculating the normalized mutual information score.")
    parser.add_argument("fixed_image", type=str, help="The fixed image.")
    parser.add_argument("moving_image", type=str, help="The moving image.")
    args = parser.parse_args()
    main(args.fixed_image, args.moving_image)


