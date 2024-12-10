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
    # sample ID is not generally a good unique ID. For 51, filepath is what you should use when working
    # with the same dataset over and over again
    if "second_play_photos" in fo.list_datasets():
        fo.delete_dataset("second_play_photos")
    return dataset_with_orig.exclude_by("filepath", first_training.values("filepath")).take(SAMPLES_TO_TAKE).clone("second_play_photos", persistent=True)

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
        # save=True,
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

def extract_orig_path(dataset):
    sample = dataset.first()
    result = sample.filepath.replace(sample.filename, "")
    return result


if __name__ == '__main__':
    print("starting")
    whole_dataset = load_data("photo_album")
    original_path = extract_orig_path(whole_dataset)
    first_training = load_data("labeled_dataset")
    candidate_data = split_again(whole_dataset,first_training)
    print("about to predict")
    results = run_predictions(candidate_data)
    
    # Convert the results list into a dict so we can iterate through the dataset rather than
    # the results. This will allow for faster saves to the dataset rather than individual sample by sample saves
    # Using the images name without the path as the key
    results_dict = {x.path[x.path.rfind("/")+1:]:x for x in results}

    # Now for each sample in the new dataset, add the prediction
    for sample in candidate_data.iter_samples(progress=True, autosave=True):
        filename = sample.filename
        res = results_dict[filename]
        predicted_class = Classification(label=res.names[res.probs.top1], confidence=round(res.probs.top1conf.item(), 2))
        sample["prediction"] = predicted_class

    # Display new dataset and hold it open with a wait()
    session = fo.launch_app(candidate_data)
    session.wait()
    print("Done")


