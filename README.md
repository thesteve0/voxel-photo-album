I got lost in the challenge of finding good classes and forgot my main goal.
My goals was to quickly find the raw images I wanted to work with for production.
This is not, given photos of dogs, which breed is it. This is, here is a general photo that could be of anything, do I care 
about it.

I only need minimal classes for that:
1. I could simply do - People vs Not people
2. Or I could do people, animal, plant, landscape, buildings

People means there is a person in the photo. Even if the people are small or singular in the picture it should be considered 
a people picture.

For the second scheme, this will be more dependent on my deciding what is the intended subject of the photo. For example,
a plant with an insect centered on it would be insect. 

THere is an open question about how to handle photos that I am just not interested in, like a picture of a box or a picture from
a doorway but just is not interesting for development. There are no people in the picture and there is not one of the classes I am interested in. I think this is basically teaching the model my concept of uninteresting. 

Doing it with the 16 classes lead to overlapping categories and unclean labeling. This in turn led to poor model performance
but for "understandble" reasons. This was not a problem with the models but a problem with data prep

Markus gave the suggestion - which is a really good one - to run this in two stages. First, train a model for people not people. Then for the not-people use that data to train a multi-class model

@@TODO I need to rename the datasets, and their references in code, to make more sense
The way to rename a dataset is
```python
import fiftyone as fo
dataset = fo.load_dataset("foo")
dataset.name = "footwo"
```

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

#### The classes I am going for
"boy"
"girl"
"man"
"woman"
"people"
"dog"
"cat"
"bird"
"insect"
"monkey"
"crustacean",
"fish"
"animal"
"plant"
"flower"
"landscape"
"architecture"
"not an animal, plant, landscape, person, or building"