import keras
import os
from keras.models import *
from keras.layers import *
from keras.datasets import cifar10
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.utils import *
from keras.applications.vgg16 import VGG16
from image_dataset_loader import load

train_data_dir = os.path.dirname(r"Dataset\TestTubes")
validation_data_dir = os.path.dirname(r"Dataset\TestTubes")

img_width, img_height = 220, 380
batch_size = 16


datagenerate=ImageDataGenerator(rescale=1. /255,
                                validation_split=0.2)
train_generator=datagenerate.flow_from_directory(train_data_dir,target_size=(img_width,img_height),
                                                 batch_size=batch_size,
                                                 subset="training",
                                                 class_mode='categorical')

validation_generator = datagenerate.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width,img_height),
                                                        batch_size=batch_size,
                                                        subset="validation",
                                                        class_mode='categorical')

print(train_generator)

def define_VGGmodel():
    model = VGG16(include_top=False, input_shape=(img_width, img_height, 3))
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(8, activation='softmax')(class1)
    model = Model(inputs=model.inputs, outputs=output)
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

model = define_VGGmodel()
model.summary()

history = model.fit(train_generator, steps_per_epoch=len(train_generator), validation_data=validation_generator, validation_steps=2, epochs=4, verbose=1, shuffle=False)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

print("Evalutation completed")

