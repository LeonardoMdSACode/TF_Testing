# CNN of cats and dogs classification, images have colors and diferent sizes
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# get data from URL
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)

# check data dir
zip_dir_base = os.path.dirname(zip_dir)
for dirpath, dirnames, filenames in os.walk(zip_dir_base):
   for dirname in dirnames:
      print(os.path.join(dirpath, dirname))

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

# understand data
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('total training cat images: ', num_cats_tr)
print("total training dog images: ", num_dogs_tr)
print("total validation cat images: ", num_cats_val)
print("total validation dog images: ", num_dogs_val)
print("Total training images: ", total_train)
print("Total validation images: ", total_val)

# Model Parameters
print("Parameters")
BATCH_SIZE = 100  # Number of examples processed b4 updating model variables thro feedback loop
IMG_SHAPE = 150  # 150x150 pixels

# Data Prep
print("Data prep")
print("ImageDataGenerator")
print(".flow_from_directory")
print("train and validation datasets")
train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator training data
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),  # (150,150)
                                                              class_mode='binary')
# Define Model
print("CNN")
print(".Sequential")
model = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(512, activation='relu'),
   tf.keras.layers.Dense(2, activation='softmax')
])

print("Compile model")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('Model Summary')
print(".sumary()")
model.summary()

print("Training Model:")
print(".fit")
EPOCHS = 1
history = model.fit(
   train_data_gen,
   steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
   epochs=EPOCHS,
   validation_data=val_data_gen,
   validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)

print("Validation Accuracy:", history.history['val_accuracy'][-1])
