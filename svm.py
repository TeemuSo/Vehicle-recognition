import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import matplotlib.pyplot as plt
import matplotlib.image

from collections import defaultdict
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

root = 'train/train/'
datagen = ImageDataGenerator()
train_data = datagen.flow_from_directory(root, class_mode='categorical', batch_size=64)

model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation='relu',
          input_shape=(256,256,3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(17, activation='softmax'))

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
model.fit_generator(train_data, steps_per_epoch=312)