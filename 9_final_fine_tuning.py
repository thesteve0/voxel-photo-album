import fiftyone as fo
import os
import tempfile
import torch
from ultralytics import YOLO

FIRST_TRAINING = "low_quality_first_labeled_dataset"
SECOND_TRAINING = ""

DATASET_NAME = 'labeled_dataset'
DEFAULT_MODEL_SIZE = "x"
DEFAULT_IMAGE_SIZE = 704
DEFAULT_EPOCHS = 12
PROJECT_NAME = 'sp_photos_yolo11'



def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

if __name__ == '__main__':
    print("starting")

    print("finished")