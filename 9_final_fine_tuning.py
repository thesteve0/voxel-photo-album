import fiftyone as fo
import os
import tempfile
import torch
from ultralytics import YOLO

FIRST_TRAINING = "first_labeled_dataset"
SECOND_TRAINING = "second_labeled_dataset"

DATASET_NAME = ''
DEFAULT_MODEL_SIZE = "l"
DEFAULT_IMAGE_SIZE = 704
DEFAULT_EPOCHS = 16
SAVE_RESULTS = True
PROJECT_NAME = 'sp_final_training_photos_yolo11'

def merge_datasets():
    first_data = fo.load_dataset(FIRST_TRAINING).clone()
    second_data = fo.load_dataset(SECOND_TRAINING)
    first_data.add_samples(second_data.view())
    return first_data
    print("Done Merging")


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
        dataset.take(round(0.3 * len(dataset))).tag_samples("test")
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
        save = SAVE_RESULTS,
        name = "output",
        exist_ok = True,
        project=project_name,
    )

    return model

if __name__ == '__main__':
    print("starting")
    merged_data = merge_datasets()
    train_classifier(
        dataset_name=merged_data.name,
        # model_size=args.model_size,
        # image_size=args.image_size,
        # epochs=args.epochs,
        project_name=PROJECT_NAME,
    )

    """
    PREP DATA
    0. check to see if the dataset we are about to make exists, if so delete it
    1. clone one of the datasets but do NOT make it persistent
    2. then add the other samples to the first dataset
    first_dataset.add_samples(second_dataset.view())
    3. Now we are ready to train using the ground_truth field
    
    TRAIN MODEL
    """

    print("finished")