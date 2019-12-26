from os import listdir
from pickle import dump
import numpy as np

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def extract_features(directory):
    # Get the InceptionV3 model trained on imagenet data
    model = InceptionV3(weights='imagenet')
    # Remove the last layer (output softmax layer) from the inception v3
    model_new = Model(model.input, model.layers[-2].output)
    # extract features from each photo in the directory
    # summarize
    print(model.summary())
    features = dict()
    # extract features from each photo
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        # Convert all the images to size 299x299 as expected by the
        img = image.load_img(filename, target_size=(299, 299))
        # Convert PIL image to numpy array of 3-dimensions
        x = image.img_to_array(img)
        # Add one more dimension
        x = np.expand_dims(x, axis=0)
        # preprocess images using preprocess_input() from inception module
        x = preprocess_input(x)
        # get features
        x = model_new.predict(x, verbose=0)
        # reshape from (1, 2048) to (2048,)
        feature = np.reshape(x, x.shape[1])
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print(len(features))
    return features


# extract features from all images
directory = ('Train_Image_Path' + 'Flicker8k_Dataset')
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('/processed_data/features.pkl', 'wb'))
