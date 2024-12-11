import fiftyone as fo
import argparse
from fiftyone import ViewField as F

""" 
This script needs to accept arguments because we don't want to do all the tasks every time we run it.
* at the end means it happens in the app, not here in code
So what we do is:
0. load in the new dataset - "second_play_photos"
1. * Go through the samples and if the prediction is incorrect then add a tag with the correct field 
2. After first pass look at all teh untagged images and make sure they are correct:
```
from fiftyone import ViewField as F
view = dataset.match(F("tags") == [])
```


5. When finished all the images, anything that has a blank tag should have the ground
truth field set = to the prediction field. 
Then we are ready to do our final training/test
Then final predictions

To do the ground_truth updates use the console
import fiftyone as fo
dataset = fo.load_dataset("second_play_photos")
session = fo.launch_app(dataset)

"""
DATASET_NAME = "second_play_photos"

if __name__ == '__main__':
    print("starting")

    dataset = fo.load_dataset(DATASET_NAME)

    parser = argparse.ArgumentParser()
    # parser.add_argument("-g", help="specify to create ground truth field", action="store_true")
    parser.add_argument("-p", help="specify to move correct predictions to ground truth field",
                        action="store_true")
    args = parser.parse_args()

    # if args.g:
    #     print("make the field")
    #     if dataset.has_field("ground_truth"):
    #         dataset.delete_sample_field("ground_truth")
    #     dataset.add_sample_field(
    #         "ground_truth",
    #         fo.EmbeddedDocumentField,
    #         embedded_doc_type=fo.Classification,
    #     )
    #     dataset.save()
    if args.p:
        # Add ground_truth label to dataset
        if dataset.has_field("ground_truth"):
            dataset.delete_sample_field("ground_truth")
        # Must initialize new `Label` values via `set_values()`
        # This basically creates a Classification for each sample in the data set which just contains an id
        dataset.set_values(
            "ground_truth",
            [fo.Classification() for _ in range(len(dataset))],
        )

        # Expression that grabs first tag if one exists, else falls back to the predicted label
        # We need an expression for set_field
        # This is basically saying "Test if tags is not empty, if that is true then return the first tag, else return the prediction label
        label_expr = (F("$tags") != []).if_else(
            F("$tags")[0],
            F("$prediction.label"),
        )

        # Apply changes
        dataset.set_field("ground_truth.label", label_expr).save()
    else:
        # print("you need to specify either -g or -p")
        print("you need to specify  -p")

    print("done")
