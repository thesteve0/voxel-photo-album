import fiftyone as fo
import argparse

""" 
This script needs to accept arguments because we don't want to do all the tasks every time we run it.
* at the end means it happens in the app, not here in code
So what we do is:
0. load in the new dataset - "second_play_photos"
1. Create a new field for ground truth
2. One at a time, go through each of the categories and just view that category
3. If there is a mislabel, then fix it in the ground truth field. 
4. Once all the images have been fixed for that category, head on to the next category
5. When finished all the categories, anything that has a blank ground truth field should have the ground
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
    parser.add_argument("-g", help="specify to create ground truth field", action="store_true")
    parser.add_argument("-p", help="specify to move correct predictions to ground truth field",
                        action="store_true")
    args = parser.parse_args()

    if args.g:
        print("make the field")
        if dataset.has_field("ground_truth"):
            dataset.delete_sample_field("ground_truth")
        dataset.add_sample_field(
            "ground_truth",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Classification,
        )
        dataset.save()
    elif args.p:
        for sample in dataset.iter_samples(autosave=True):
            if sample["ground_truth"] is None:
                sample["ground_truth"] = sample["prediction"]

    else:
        print("you need to specify either -g or -p")

    print("done")

