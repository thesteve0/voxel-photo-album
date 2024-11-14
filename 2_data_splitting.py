import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from sympy.core.random import shuffle

dataset_name = "photo_album"

# This just takes a sample of the original dataset and creates
# a new dataset which is has a random 300 samples.
# For real work we probably would just want to tag this data. Something like training,
# test, validation

if __name__ == "__main__":
    print("starting")
    dataset = fo.load_dataset(dataset_name)
    # A DatasetView is a shallow copy of the original dataset.This means,
    # if your dataset is persisted, any data or schema changes to the view will be persisted to the original dataset.
    # If you don't want to affect the original dataset you must make the view a clone of the original data.
    shuffle_view = dataset.shuffle()
    training_view = shuffle_view[0:300].clone("play_photos", persistent=True)
    training_view.save()

    print("finished")