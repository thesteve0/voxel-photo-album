import fiftyone as fo
import fiftyone.zoo as foz
from imageio.plugins.ffmpeg import download

dataset_name = "photo_album"
# model = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")
# samples.compute_embeddings(model, embeddings_field="embeddings")

if __name__ == "__main__":



    print(fo.list_datasets())
    downloaded_models = foz.list_downloaded_zoo_models()
    dataset = fo.load_dataset(dataset_name)

    # Eventually this will be a loop to calculate and store embeddings
    model = foz.load_zoo_model('dinov2-vitb14-torch')
    dataset.compute_embeddings(model, embeddings_field="densenet121-imagenet                                                ")
    session = fo.launch_app(dataset)
    session.wait()

    print("done")