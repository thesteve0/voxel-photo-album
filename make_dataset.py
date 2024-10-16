import fiftyone as fo
from fiftyone.core.metadata import compute_metadata

# this is the name the for the dataset inside 51
name = "my-dataset"

# this is the path where the images are located
dataset_dir = "/home/spousty/data/ai-ready-images"



# according to the doc it will read any image that has an image mim-type and ignore the rest
# it will go recuresively down through directories
def import_and_create():
 # Create the dataset
 dataset = fo.Dataset.from_dir(
  dataset_dir=dataset_dir,
  dataset_type=fo.types.ImageDirectory,
  compute_metadata=True,
  name=name,
 )

def start_fiftyone():
  dataset = fo.load_dataset(name)
  session = fo.launch_app(dataset)
  session.wait(-1)

  # To delete the dataset
  # https://docs.voxel51.com/user_guide/using_datasets.html#deleting-a-dataset

if __name__ == "__main__":
 print("reading in data")
 import_and_create()
 start_fiftyone()
 print("done")
