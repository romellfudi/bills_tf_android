# !rm -rf bills_photos/ __MACOSX/
# !curl -vLJO -H 'Accept: application/octet-stream' https://api.github.com/repos/romellfudi/bills_tf_android/releases/assets/26118074  -u "contactboosttag:254aa92f4c88b57bdbc42070fbd0c66e58d00121" 
# !unzip bills_photos.zip
# !find . -name '*.xml' -delete

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

os.chdir('bills_photos')
if os.path.isdir('train/10') is False:
    os.makedirs('train/10')
    os.makedirs('train/20')
    os.makedirs('train/50')
    os.makedirs('train/100')
    os.makedirs('valid/10')
    os.makedirs('valid/20')
    os.makedirs('valid/50')
    os.makedirs('valid/100')
    os.makedirs('test/10')
    os.makedirs('test/20')
    os.makedirs('test/50')
    os.makedirs('test/100')

    for i in random.sample(glob.glob('Billetes 10/*.JPG'), 120):
        shutil.move(i, 'train/10')      
    for i in random.sample(glob.glob('Billetes 20/*.JPG'), 120):
        shutil.move(i, 'train/20')   
    for i in random.sample(glob.glob('Billetes 50/*.JPG'), 120):
        shutil.move(i, 'train/50')   
    for i in random.sample(glob.glob('Billetes 100/*.JPG'), 120):
        shutil.move(i, 'train/100')
    for i in random.sample(glob.glob('Billetes 10/*.JPG'), 60):
        shutil.move(i, 'valid/10')   
    for i in random.sample(glob.glob('Billetes 20/*.JPG'), 60):
        shutil.move(i, 'valid/20')   
    for i in random.sample(glob.glob('Billetes 50/*.JPG'), 60):
        shutil.move(i, 'valid/50')        
    for i in random.sample(glob.glob('Billetes 100/*.JPG'), 60):
        shutil.move(i, 'valid/100')
    for i in random.sample(glob.glob('Billetes 10/*.JPG'), 20):
        shutil.move(i, 'test/10')  
    for i in random.sample(glob.glob('Billetes 20/*.JPG'), 20):
        shutil.move(i, 'test/20')  
    for i in random.sample(glob.glob('Billetes 50/*.JPG'), 20):
        shutil.move(i, 'test/50')      
    for i in random.sample(glob.glob('Billetes 100/*.JPG'), 20):
        shutil.move(i, 'test/100')

os.chdir('../')


train_path = 'bills_photos/train'
valid_path = 'bills_photos/valid'
test_path = 'bills_photos/test'

train_batches = ImageDataGenerator(rotation_range=90) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['10', '20', '50', '100'], batch_size=10)
valid_batches = ImageDataGenerator(rotation_range=90) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['10', '20', '50', '100'], batch_size=10)
test_batches = ImageDataGenerator(rotation_range=90) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['10', '20', '50', '100'], batch_size=10, shuffle=False)

# imgs, labels = next(train_batches)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=4, activation='softmax')
])

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=10,
    verbose=2
)
print(test_batches.class_indices)

model.save('keras_model.h5')


import tensorflow.keras as keras
import tensorflow 
from tensorflow.keras.models import Sequential, load_model
print(tensorflow.__version__)
print(keras.__version__)
# model = load_model('image_generation_model.h5')
# print(" * Model loaded!")
model = load_model('keras_model.h5')
print(" * Model loaded!")