import argparse
import os
import numpy as np
import SimpleITK as sitk
import torch
import sys
import yaml
from monai.networks.nets import UNet
from monai.inferers import SlidingWindowInfererAdapt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.utils import image_to_array, array_to_image
from src.post_process import post_processing_retina
from src.model.retina_UNet import Retina_UNet


def normalize_image(image: np.ndarray):
    """
    Normalize the image to have zero mean and unit variance
    Args:
        image: numpy array of the image
    Returns:
        image: numpy array of the normalized image
    """
    image = image.astype(np.float64)
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / std

    # Intensity normalization to [-1, 1]
    min_value = np.min(image)
    max_value = np.max(image)
    image = (image - min_value) / (max_value - min_value)
    image = image * 2 - 1
    image = np.clip(image, -1, 1)

    return image

def load_UNet_model(model_path: str, config, cpu=False):
    """
    Load the UNet model
    Args:
        model_path: path to the model
    Returns:
        model: the UNet model
    """
    model = UNet(
        spatial_dims=config["model"]["spatial_dims"],
        in_channels=1,
        out_channels=2,
        channels=config["model"]["features"],
        strides=config["model"]["strides"],
        dropout=config["model"]["dropout"],
        norm=config["model"]["norm"],
        act=config["model"]["activation"],
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') if cpu else torch.device('cuda')))
    model.eval()
    return model

def load_RetinaUNet_model(model_path: str, config, cpu=False):
    """
    Load the RetinaUNet model
    Args:
        model_path: path to the model
    Returns:
        model: the RetinaUNet model
    """
    model = Retina_UNet(1, 2, 1, config, mode='test')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') if cpu else torch.device('cuda')))
    model.eval()
    return model

def inference_2D(model, image: np.ndarray, cpu=False):
    """
    Perform inference on a 2D image
    Args:
        model: the model
        image: the 2D image
    Returns:
        cortical: the predicted cortical mask
        trabecular: the predicted trabecular mask
    """
    image = image.astype(np.float32)

    image = torch.tensor(image)
    image = torch.nn.functional.pad(image, (0,0,0,0,1,1), mode='reflect')
    image = image

    if not cpu:
        image = image.to("cuda")
    with torch.no_grad():
        predictor = SlidingWindowInfererAdapt(roi_size=[3,256,256], sw_batch_size=1, overlap=0.25, mode="gaussian", progress=False)
        cortical, trabecular = [],[]
        for i in range(1, image.shape[1]-1):
            pred = model(image[i-1:i+1])
            cortical.append(pred[0].squeeze())
            trabecular.append(pred[1].squeeze())
        cortical = torch.stack(cortical, dim=0)
        trabecular = torch.stack(trabecular, dim=0)

    cortical = torch.nn.functional.sigmoid(cortical).cpu().squeeze().numpy()
    trabecular = torch.nn.functional.sigmoid(trabecular).cpu().squeeze().numpy()

    return cortical, trabecular


def inference_3D(model, image, cpu):
    """ Perform inference on a 3D image

    Args:
        model: the model
        image: the 3D image
    Returns:
        cortical: the predicted cortical mask
        trabecular: the predicted trabecular mask
    """

    image = image.astype(np.float32)
    image = torch.tensor(image)
    image = image.unsqueeze(0)

    if not cpu:
        image = image.to("cuda")
    
    predictor = SlidingWindowInfererAdapt(roi_size=[64,64,64], sw_batch_size=1, overlap=0.25, mode="gaussian", progress=False)
    with torch.no_grad():
        pred_mask = predictor(image, model)
    # pred_mask = sliding_window_inference(input, roi_size=[64,64,64], sw_batch_size=8, predictor=model, overlap=0.25, mode="gaussian")
    pred_mask = torch.nn.functional.sigmoid(pred_mask).to('cpu').detach().squeeze().numpy()

    return pred_mask[0], pred_mask[1]


def main(image_path, model_path, output_path, model_type, config, cpu=False):
    """
    Perform inference on the image
    Args:
        image_path: path to the image
        model_path: path to the model
        output_path: path to save the output
        config_path: path to the configuration file
    """
    assert model_type in ["3d", "2d", "retina"]

    image = sitk.ReadImage(image_path)
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()
    image = sitk.GetArrayFromImage(image)
    image = normalize_image(image)

    if model_type == "3d" or model_type == "2d":
        model = load_UNet_model(model_path, config, cpu)
    else:
        model = load_RetinaUNet_model(model_path, config, cpu)
    
    if model_type == "3d" or model_type == "retina":
        cortical, trabecular = inference_3D(model, image, cpu)
    else:
        cortical, trabecular = inference_2D(model, image, cpu)

    cortical = post_processing_retina(image, cortical, trabecular)

    cortical = array_to_image(cortical, spacing, origin, direction)
    trabecular = array_to_image(trabecular, spacing, origin, direction)

    sitk.WriteImage(cortical, os.path.join(output_path, "cortical_mask.nii.gz"))
    sitk.WriteImage(trabecular, os.path.join(output_path, "trabecular_mask.nii.gz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=False, default="config/segmentation_config.yaml")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    args.model_type = args.model_type.lower()

    config = yaml.safe_load(open(args.config_path, "r"))
    if args.model_type == "2d":
        config["input_shape"] = [3, 256, 256]
    main(args.image_path, args.model_path, args.output_path, args.model_type, config, args.cpu)
