# voxel-photo-album

Here are the general steps we want to accomplish
1. photos -> dataset
1. dataset to datasetview - some for fitting, some for test, and some for validation
1. dataset -> embeddings
   3. https://docs.voxel51.com/model_zoo/index.html
1. show how different transformers embed
1. Talk about how to sample that space and what happens if you exclude and area
1. Look for out of focus images (maybe also low contrast ones)
1. Now do the predictions with each embedding and see how they do
1. Now fine train at least one if not two and compare them both for ease and for accuracy

## Order to run things
1. First run make_dataset.py to make the dataset in FiftyOne
2. Now run make_embeddings.py to create all the embeddings