
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
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

from PIL import Image
import numpy as np 

os.chdir('bills_photos')
if os.path.isdir('test/10') is False:
    os.makedirs('test/10')
    os.makedirs('test/20')
    os.makedirs('test/50')
    os.makedirs('test/100') 
    for i in random.sample(glob.glob('Billetes 10/*.JPG'), 190):
        shutil.move(i, 'test/10')      
    for i in random.sample(glob.glob('Billetes 20/*.JPG'), 190):
        shutil.move(i, 'test/20')   
    for i in random.sample(glob.glob('Billetes 50/*.JPG'), 190):
        shutil.move(i, 'test/50')   
    for i in random.sample(glob.glob('Billetes 100/*.JPG'), 190):
        shutil.move(i, 'test/100') 

os.chdir('../')

test_path = 'bills_photos/test'

valid_batches = ImageDataGenerator(rotation_range=90) \
    .flow_from_directory(directory=test_path,  target_size=(200,200), classes=['10', '20', '50', '100'], batch_size=5)

model = load_model('keras_model.h5')

result = model.evaluate(x=valid_batches)
print(dict(zip(model.metrics_names, result)))

i,l = next(valid_batches)
print(type(i[0]),l[0])

img = Image.fromarray(i[0], 'RGB')
img.save('my1%s.png'% (l[0]))
img = Image.fromarray(i[1], 'RGB')
img.save('my2%s.png'% (l[1]))
img = Image.fromarray(i[2], 'RGB')
img.save('my3%s.png'% (l[2]))
img = Image.fromarray(i[3], 'RGB')
img.save('my4%s.png'% (l[3]))