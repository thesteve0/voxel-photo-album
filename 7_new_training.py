import fiftyone as fo
import tempfile
import torch
from safetensors.torch import save_model
from ultralytics import YOLO

"""
We have trained the model on the < 5% of the original data. 
Now we are going to take this newly trained model and train it some more on another chunk of the data.
We are going to have the new model predict the new images and then correct it to the ground truth again.

First step is going to be splitting out more data that is not in the first training set. 
Then run the new model over it producing the new predictions
"""

# This is the number we need to make our total sampled data = 10%
SAMPLES_TO_TAKE = 454
IMAGE_SIZE = 704
MODEL_LOCATION = "/home/spousty/git/voxel-photo-album/sp_photos_yolo11/train6/weights/best.pt"

def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_data(datasetname) -> fo.Dataset:
    return fo.load_dataset(datasetname)


def split_again(dataset_with_orig, first_training) -> fo.Dataset:
    # Take a random sample from the original data, excluding the first set of sample data
    # We
    return dataset_with_orig.exclude(first_training).take(SAMPLES_TO_TAKE)

def run_predictions(dataset):
    model = YOLO(MODEL_LOCATION)  # load a custom model
    # naive_model = YOLO(f"yolo11x-cls.pt")

    #"export" our sample images to disk - symlink
    data_dir = tempfile.mkdtemp()
    dataset.export(export_dir=data_dir, dataset_type=fo.types.ImageDirectory,export_media="symlink")

    # Predict with the model
    results = model(
        source=data_dir,
        device=get_torch_device(),
        imgsz=IMAGE_SIZE,
        stream=True,
        save=True,
        project="predictions_round2",
        name="write_something"
    )

    # naive_results = naive_model(
    #     source=data_dir,
    #     device=get_torch_device(),
    #     imgsz=IMAGE_SIZE,
    #     stream=True,
    #     save=True,
    #     project="naive_yolo",
    #     name="write out the files"
    # )

    # return [results, naive_results]
    return results


if __name__ == '__main__':
    print("starting")
    whole_dataset = load_data("photo_album")
    first_training = load_data("labeled_dataset")
    candidate_data = split_again(whole_dataset,first_training)
    print("about to predict")
    results = run_predictions(candidate_data)
    for r in results[0]:
        # we save the data to a 51 dataset. Now that we are done exploring, I think we might want to clone above.
       r.save()
    # for r2 in results[1]:
    #     r2.save()
    print("Done")


