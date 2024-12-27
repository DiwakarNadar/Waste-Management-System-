import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.activations import relu,sigmoid, softmax
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model

# Load the model

import joblib


class_names = joblib.load('class_names.joblib')
def create_model():
    model = load_model('RealWasteFineTuning.hdf5')
    return model


# def load_and_prep_image(filename, img_shape=224, scale=True):
#   img = tf.io.read_file(filename)

#   img = tf.io.decode_image(img, channels=3)

#   img = tf.image.resize(img, [img_shape, img_shape])

#   if scale:
#     return img/255.
#   else:
#     return img

# def prediction(model):
#     pred_prob = model.predict(tf.expand_dims(img, axis=0))
#     pred_class = class_names[pred_prob.argmax()]
#     print(pred_class,":",pred_prob.max())