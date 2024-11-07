import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

from imageio.plugins.ffmpeg import download

dataset_name = "training_data"
# model = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
# samples.compute_embeddings(model, embeddings_field="embeddings")

if __name__ == "__main__":

    training_view = fo.load_dataset("training_data")

    model = foz.load_zoo_model('resnet101-imagenet-torch')
    fob.compute_visualization(training_view, model=model, embeddings="sresnet101_imagenet",
                              brain_key="resnet101_imagenet_embed")

    model2 = foz.load_zoo_model("vgg11-bn-imagenet-torch")
    fob.compute_visualization(training_view, model=model2, embeddings="fvgg11_bn", brain_key="vgg11_bn_embed")

    session = fo.launch_app(dataset)
    session.wait()

    print("done")