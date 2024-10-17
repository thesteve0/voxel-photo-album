from pathlib import Path

import fiftyone as fo
import exifread
from PIL import Image
import exiftool


# this is the name the for the dataset inside 51
name = "my-dataset"

# this is the path where the images are located
source_image_dir = "/home/spousty/data/ai-ready-images"



# according to the doc it will read any image that has an image mim-type and ignore the rest
# it will go recuresively down through directories
def simple_import_and_create():
    # Create the dataset. This one works with a directory of images if we don't care about any of the EXIF data.
    # Since we want EXIF data
    dataset = fo.Dataset.from_dir(
        dataset_dir=source_image_dir,
        dataset_type=fo.types.ImageDirectory,
        compute_metadata=True,
        name=name,
    )

# 'Image Model'
# 'EXIF ExposureTime'
# 'EXIF FNumber'
# 'EXIF DateTimeOriginal'
# 'EXIF ShutterSpeedValue'
# 'EXIF ApertureValue'
# 'EXIF Flash'
# 'EXIF SensingMethod'
# 'EXIF SubjectDistance'
# 'EXIF FocalLength'

def import_and_create_with_metadata():
    exif_fields = ["Image Model", "EXIF ExposureTime", "EXIF FNumber", "EXIF DateTimeOriginal", "EXIF ShutterSpeedValue",
                   "EXIF ApertureValue", "EXIF Flash", "EXIF SensingMethod", "EXIF SubjectDistance", "EXIF FocalLength"]
    path = Path(source_image_dir)
    for sample in path.rglob('*.JPG'):
        f = open(sample, 'rb')
        tags = exifread.process_file(f)
        for field in exif_fields:
            if field in exif_fields and tags.get(field) is not None:
                print("got one: " + field + " : " + str(tags[field]) )
        print("maybe")



def start_fiftyone():
    dataset = fo.load_dataset(name)
    session = fo.launch_app(dataset)
    session.wait(-1)

    # To delete the dataset
    # https://docs.voxel51.com/user_guide/using_datasets.html#deleting-a-dataset

if __name__ == "__main__":
    print("reading in data")
    import_and_create_with_metadata()
    start_fiftyone()
    print("done")
