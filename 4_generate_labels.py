from http.cookiejar import debug

import fiftyone as fo
import fiftyone.zoo as foz


if __name__ == '__main__':
    dataset = fo.load_dataset("play_photos")
    clip = foz.load_zoo_model(
        "clip-vit-base32-torch",
        text_prompt="A photo of a",
        classes= ["boy", "girl", "man", "woman", "people", "dog", "cat", "bird", "insect", "monkey", "crustacean",
                  "fish", "animal", "plant", "flower", "landscape", "architecture", "not an animal, plant, landscape, person, or building"])
    # alexnet = foz.load_zoo_model("alexnet-imagenet-torch")
    # dense201 = foz.load_zoo_model("densenet201-imagenet-torch")
    # fasterrcnn = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
    # yoloseg = foz.load_zoo_model("yolo11x-seg-coco-torch")

    dataset.apply_model(clip, label_field="default_prediction")
    dataset.set_values("ground_truth", fo.Classification(label=str(dataset.values("default_prediction.label"))))
    # dataset.apply_model(dense201, label_field="dense201")
    # dataset.apply_model(alexnet, label_field="alexnet")
    # dataset.apply_model(fasterrcnn, label_field="faster_rcnn")
    # dataset.apply_model(yoloseg, label_field="yolo_seg")

    session = fo.launch_app(dataset)
    session.wait()
