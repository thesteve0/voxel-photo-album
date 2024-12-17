import fiftyone as fo
import tempfile
import torch
from fiftyone import Classification
from ultralytics import YOLO

"""
We have trained the model on the < 5% of the original data. 
Now we are going to take this newly trained model and train it some more on another chunk of the data.
We are going to have the new model predict the new images and then correct it to the ground truth again.

First step is going to be splitting out more data that is not in the first training set. 
Then run the new model over it producing the new predictions
"""

# This is the number we need to make our total sampled data = 10%
FIRST_TRAINING = "first_labeled_dataset"
SECOND_TRAINING = "second_labeled_dataset"
FINAL_OUTPUT = "final_predicted_photos"
IMAGE_SIZE = 704
MODEL_LOCATION = "/home/spousty/git/voxel-photo-album/sp_final_training_photos_yolo11/output/weights/best.pt"


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
    # sample ID is not generally a good unique ID. For 51, filepath is what you should use when working
    # with the same dataset over and over again
    if "second_play_photos" in fo.list_datasets():
        fo.delete_dataset("second_play_photos")
    return dataset_with_orig.exclude_by("filepath", first_training.values("filepath"))


def run_predictions(dataset):
    model = YOLO(MODEL_LOCATION)  # load a custom model
    # naive_model = YOLO(f"yolo11x-cls.pt")

    # "export" our sample images to disk - symlink
    data_dir = tempfile.mkdtemp()
    dataset.export(export_dir=data_dir, dataset_type=fo.types.ImageDirectory, export_media="symlink")

    # Predict with the model
    results = model(
        source=data_dir,
        device=get_torch_device(),
        imgsz=IMAGE_SIZE,
        stream=True,
        # save=True,
        project="final_predictions"
    )

    return results


def extract_orig_path(dataset):
    sample = dataset.first()
    result = sample.filepath.replace(sample.filename, "")
    return result

def merge_all_training_data():
    first_data = fo.load_dataset(FIRST_TRAINING).view().clone()
    second_data_view = fo.load_dataset(SECOND_TRAINING).view()
    first_data.add_samples(second_data_view)
    return first_data


if __name__ == '__main__':
    print("starting")
    whole_dataset = load_data("photo_album")
    original_path = extract_orig_path(whole_dataset)
    all_training_data = merge_all_training_data()
    if FINAL_OUTPUT in fo.list_datasets():
        fo.delete_dataset(FINAL_OUTPUT)
    remaining_photos = split_again(whole_dataset, all_training_data).clone(FINAL_OUTPUT, persistent=True)
    print("about to predict")
    results = run_predictions(remaining_photos)

    # Convert the results list into a dict so we can iterate through the dataset rather than
    # the results. This will allow for faster saves to the dataset rather than individual sample by sample saves
    # Using the images name without the path as the key
    # This is causing an OOM kill - I think there are too many objects in each results object
    results_dict = {}
    result = {}
    for x in results:
        file_name = x.path[x.path.rfind("/") + 1:]
        label = x.names[x.probs.top1]
        confidence = float(round(x.probs.top1conf.item(),2))
        result = {"label": label, "confidence": confidence}
        results_dict[file_name] = result

    # results_dict = {: x for x in results}

    # Now for each sample in the new dataset, add the prediction
    for sample in remaining_photos.iter_samples(progress=True, autosave=True):
        filename = sample.filename
        res = results_dict[filename]
        predicted_class = Classification(label=res["label"],
                                         confidence=res["confidence"])
        sample["prediction"] = predicted_class

    # Display new dataset and hold it open with a wait()
    #session = fo.launch_app(remaining_photos)
    #session.wait()
    print("Done")


