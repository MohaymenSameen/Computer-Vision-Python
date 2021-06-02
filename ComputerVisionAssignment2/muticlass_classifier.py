import keras
from keras.models import *
from keras.layers import *
from keras.datasets import cifar10
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.utils import *
from keras.applications.vgg16 import VGG16


# creating training and testing datset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
classification = ['BD-Vacutainer', 'EDTA', 'Monovette', 'Plain', 'SGS', 'Vacuette', 'Vacutest', 'Venosafe']

# lets print data type of our dataset
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))
