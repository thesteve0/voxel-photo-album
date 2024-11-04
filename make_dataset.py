from doctest import debug
from pathlib import Path

import fiftyone as fo
import exifread


# this is the name the for the dataset inside 51
name = "photo_album"

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

def import_and_create_with_fields():

    dataset = fo.Dataset(name, overwrite=True, persistent=True)

    relevant_exif_fields = ["Image Model", "EXIF ExposureTime", "EXIF FNumber", "EXIF DateTimeOriginal", "EXIF ShutterSpeedValue",
                   "EXIF ApertureValue", "EXIF Flash", "EXIF SensingMethod", "EXIF SubjectDistance", "EXIF FocalLength"]
    path = Path(source_image_dir)
    samples = []

    for sample_file in path.rglob('*.JPG'):
        exif_fields = {}
        f = open(sample_file, 'rb')
        try:
            tags = exifread.process_file(f, details=False)
            for field in relevant_exif_fields:
                if field in relevant_exif_fields and tags.get(field) is not None:
                    exif_fields[field] = str(tags.get(field))
            # We have built our dict of exif info. Time to make the sample
            sample = fo.Sample(filepath = sample_file, **exif_fields)
            # Now add that sample to the list of samples
            samples.append(sample)
        except:
            print("The file: " + str(sample_file) +  " the follow exif data threw an exception" + str(tags))
            pass
        f.close()

    # We have gone through all the files and created a List of samples. Time to add them to the dataset
    dataset.add_samples(samples)
    dataset.save()



def start_fiftyone():
    dataset = fo.load_dataset(name)
    session = fo.launch_app(dataset)
    session.wait(-1)

    # To delete the dataset
    # https://docs.voxel51.com/user_guide/using_datasets.html#deleting-a-dataset

if __name__ == "__main__":
    print("reading in data")
    import_and_create_with_fields()
    start_fiftyone()
    print("done")
