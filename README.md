# DeepFemurSegmentation

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







