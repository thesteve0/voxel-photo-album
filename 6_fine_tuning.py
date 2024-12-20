# import argparse
import os
import tempfile
import torch
from ultralytics import YOLO
import fiftyone as fo
from wandb.sdk.verify.verify import PROJECT_NAME

"""
All of this (except the data specific part) is from Jacob M

Ultralytics YOLOv8*-cls model training script
for generating confidence-based noise labels for a dataset.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|

Requires `ultralytics` and `fiftyone>=0.25.0` to be installed.
---------------
Steve says: 
"//I was going to to start with the imgsz =640x640 by resizing to 860x860 and then crop to 640x640//. I can't
do this because the image size below is just a magic number we set with the Yolo library. It says it 
will just resize to that size but doesn't tell us how. So instead, to start with, I am just setting this to 640.
I am also setting the model size to nano, but will bump it up over time. Settled on X - it doesn't take much longer and does better
"""

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


def train_classifier(
        dataset_name=None,
        model_size=DEFAULT_MODEL_SIZE,
        image_size=DEFAULT_IMAGE_SIZE,
        epochs=DEFAULT_EPOCHS,
        project_name="mislabel_confidence_noise",
        gt_field="ground_truth",
        train_split=None,
        test_split=None,
        **kwargs
):

    # settings.update({"wandb": False})
    if dataset_name:
        dataset = fo.load_dataset(dataset_name)
        dataset.take(0.2 * len(dataset)).tag_samples("test")
        dataset.match_tags("test", bool=False).tag_samples("train")
        train = dataset.match_tags("train")
        test = dataset.match_tags("test")
    else:
        train = train_split
        test = test_split

    if model_size is None:
        model_size = "s"
    elif model_size not in ["n", "s", "m", "l", "x"]:
        raise ValueError("model_size must be one of ['n', 's', 'm', 'l', 'x']")

    splits_dict = {
        "train": train,
        "val": test,
        "test": test,
    }

    data_dir = tempfile.mkdtemp()

    for key, split in splits_dict.items():
        split_dir = os.path.join(data_dir, key)
        os.makedirs(split_dir)
        split.export(
            export_dir=split_dir,
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            label_field=gt_field,
            export_media="symlink",
        )

    # Load a pre-trained YOLOv8 model for classification
    model = YOLO(f"yolo11{model_size}-cls.pt")

    # Train the model
    model.train(
        data=data_dir,  # Path to the dataset
        epochs=epochs,  # Number of epochs
        imgsz=image_size,  # Image size
        device=get_torch_device(),
        project=project_name,
    )

    return model


def main():
    if fo.__version__ < "0.25.0":
        raise ValueError("Please upgrade to the latest version of FiftyOne")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_name", type=str, required=True)
    # parser.add_argument("--model_size", type=str, default=None)
    # parser.add_argument("--image_size", type=int, default=128)
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--project_name", type=str, default="mislabel_confidence_noise")
    # args = parser.parse_args()

    train_classifier(
        dataset_name=DATASET_NAME,
        # model_size=args.model_size,
        # image_size=args.image_size,
        # epochs=args.epochs,
        project_name=PROJECT_NAME,
    )


if __name__ == "__main__":
    main()
