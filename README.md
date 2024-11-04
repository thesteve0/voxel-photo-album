# voxel-photo-album

Here are the general steps we want to accomplish
1. photos -> dataset
2. dataset -> embeddings
   3. https://docs.voxel51.com/model_zoo/index.html
3. show how different transformers embed
4. Talk about how to sample that space and what happens if you exclude and area
5. Look for out of focus images (maybe also low contrast ones)
6. Now do the predictions with each embedding and see how they do
7. Now fine train at least one if not two and compare them both for ease and for accuracy

## Order to run things
1. First run make_dataset.py to make the dataset in FiftyOne
2. Now run make_embeddings.py to create all the embeddings