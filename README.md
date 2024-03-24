# DeepFemurSegmentation
### Abstract
In this work, we seek to validate and segment femur images obtained from the
Photon-counting computed tomography (PCCT) scanner. Using the established
high-resolution peripheral quantitative computed tomography (HR-pQCT) im-
ages as a ground truth. We use a U-Net architecture to segment the trabecular
and cortical bone compartments. Further, introduce an auxiliary task to the U-
Net architecture, to predict the HR-pQCT images from the PCCT images. We
find that the U-Net architecture can segment the trabecular and cortical bone
with a high accuracy. Our findings indicate that the auxiliary task does not im-
prove the segmentation of the trabecular and cortical bone. We conclude that the
PCCT images can be used to extract valid cortical and trabecular bone regions
in an automated fashion
## Usage

### Installation
Create a new conda environment using python 3.11 
`conda create -n femur python=3.11`
Activate the environment and install the required packages:
`pip install -r ./requirements.txt`
depending on your Cuda version some changes might be necessary.

### Prediction
We provide model weights here.
Then the full model pipeline can be run using the `inference.py` script.
With the environment active use `python scripts/inference.py --image_path PATH_TO_IMAGE_FILE --model_path PATH_TO_MODEL_PTH --model_type [3D, 2D, RETINA]`







