# CNN of cats and dogs classification, images have colors and diferent sizes
# APPLYING IMAGE AUGMENTATION AND DROPOUT TO INCREASE ACCURACY AND DECREASE VALIDATION LOSS
from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

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

print("total training cat images: ", num_cats_tr)
print("total training dog images: ", num_dogs_tr)
print("total validation cat images: ", num_cats_val)
print("total validation dog images: ", num_dogs_val)
print("Total training images: ", total_train)
print("Total validation images: ", total_val)

# Model Parameters
print("Parameters")
BATCH_SIZE = 128  # Number of examples processed b4 updating model variables thro feedback loop
IMG_SHAPE = 150  # 150x150 pixels

# DATA AUGMENTATION
print("Data Augmentation")


# function to plot imgs
def plotImages(images_arr):
   fig, axes = plt.subplots(1, 5, figsize=(20, 20))
   axes = axes.flatten()
   for img, ax in zip(images_arr, axes):
      ax.imshow(img)
   plt.tight_layout()
   plt.show()


# Flipping the image horizontally
image_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE, IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)
print("Flipping imgs")

# Rotating the image
image_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE, IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)
print("Rotated imgs")

# Applying Zoom
image_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.5)
train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE, IMG_SHAPE))
augmented_images =  [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)
print("Zoom Applied")

# Join everything
image_gen_train = ImageDataGenerator(
   rescale=1. / 255,
   rotation_range=20,
   width_shift_range=0.2,
   height_shift_range=0.2,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   fill_mode='nearest')
train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE, IMG_SHAPE),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

# Validation Dataset Generator
image_gen_val = ImageDataGenerator(rescale=1. / 255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=validation_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')

# Define Model
print("CNN")
print(".Sequential")
model = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.layers.Flatten(),

   tf.keras.layers.Dense(512, activation='relu'),
   tf.keras.layers.Dropout(0.05),
   tf.keras.layers.Dense(1, activation='sigmoid') # 1=0.789 # 2 = 0.817
])


print("Compile model")
model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('Model Summary')
print(".sumary()")
model.summary()

print("Training Model:")
print(".fit")
CALLBACKS = [    EarlyStopping(monitor='accuracy', mode='max', baseline=0.9),    EarlyStopping(monitor='val_accuracy', mode='max', baseline=0.8)]

EPOCHS = 200
history = model.fit(
   train_data_gen,
   steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
   epochs=EPOCHS,
   validation_data=val_data_gen,
   validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))),
   callbacks=[CALLBACKS]
)
if (history.history['accuracy'][-1] >= 0.9) and (history.history['val_accuracy'][-1] >= 0.8):
    print('Training stopped as both training accuracy and validation accuracy have reached desired thresholds.')


print("Validation Accuracy:", history.history['val_accuracy'][-1])

# Epoch 200/200
# 16/16 [==============================] - 9s 537ms/step - loss: 0.2393 - accuracy: 0.9015 - val_loss: 0.4004 - val_accuracy: 0.8440
# Validation Accuracy: 0.843999981880188

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
