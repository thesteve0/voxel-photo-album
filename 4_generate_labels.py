from http.cookiejar import debug

import fiftyone as fo
import fiftyone.zoo as foz

if __name__ == '__main__':
    dataset = fo.load_dataset("play_photos")

    # Clean up from previous runs
    if "labeled_dataset" in fo.list_datasets():
        fo.delete_dataset("labeled_dataset")

    clip = foz.load_zoo_model(
        "clip-vit-base32-torch",
        text_prompt="A photo of a",
        classes= ["boy", "girl", "man", "woman", "people", "dog", "cat", "bird", "insect", "monkey", "crustacean",
                  "fish", "animal", "plant", "flower", "landscape", "architecture", "not an animal, plant, landscape, person, or building"])
    # alexnet = foz.load_zoo_model("alexnet-imagenet-torch")
    # dense201 = foz.load_zoo_model("densenet201-imagenet-torch")
    # fasterrcnn = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
    #yoloseg = foz.load_zoo_model("yolo11x-seg-coco-torch")

    dataset.apply_model(clip, label_field="prediction")
    # dataset.apply_model(dense201, label_field="dense201")
    # dataset.apply_model(alexnet, label_field="alexnet")
    # dataset.apply_model(fasterrcnn, label_field="faster_rcnn")
    #dataset.apply_model(yoloseg, label_field="yolo_seg")

    #Alright time to make our dataset with cleaned labels
    labeled_dataset = dataset.clone(name="labeled_dataset", persistent=True)
    labeled_dataset.rename_sample_field("prediction", "ground_truth")
    labeled_dataset.set_field("ground_truth.detections.confidence", None).save()

    # Now time to go to 5_clean_ground_truth

    session = fo.launch_app(dataset)
    session.wait()
