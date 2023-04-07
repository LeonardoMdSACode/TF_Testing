# Image Classification CNN, with Data Augmentation tecniques
import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Loading
print(("Data Loading"))
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

# Creating labels
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# prints the total number of flowers imgs for each type
for cl in classes:
   img_path = os.path.join(base_dir, cl)
   images = glob.glob(img_path + '/*.jpg')
   print("{}: {} Images".format(cl, len(images)))
   num_train = int(round(len(images)*0.8))
   train, val = images[:num_train], images[num_train:]

   for t in train:
      dst_dir = os.path.join(base_dir, 'train', cl)
      dst_file = os.path.join(dst_dir, os.path.basename(t))
      if not os.path.exists(dst_file):
         if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
         shutil.move(t, dst_file)
      else:
         print(f'Skipping {t}. File {dst_file} already exists.')

   for v in val:
      dst_dir = os.path.join(base_dir, 'val', cl)
      dst_file = os.path.join(dst_dir, os.path.basename(t))
      if not os.path.exists(dst_file):
         if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
         shutil.move(t, dst_file)
      else:
         print(f'Skipping {v}. File {dst_file} already exists.')

round(len(images)*0.8)
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

batch_size = 45
IMG_SHAPE = 150

print("Apply Random Horizontal Flip")
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE)
                                                )
def plotImages(images_arr):
   fig, axes = plt.subplots(1, 5, figsize=(20, 20))
   axes = axes.flatten()
   for img, ax in zip(images_arr, axes):
      ax.imshow(img)
   plt.tight_layout()
   plt.show()
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

print("Apply Random Rotation")
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE, IMG_SHAPE))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

print("Apply Random Zoom")
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE, IMG_SHAPE)
                                                )
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

print("Put it all Together")
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
train_data_gen = image_gen_train.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE),
                                                class_mode='sparse'
                                                )
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

print("Create aData Generator for the Validation Set")
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='sparse')

print("Create CNN")
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])

base_model = tf.keras.applications.ResNet50V2(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(IMG_SHAPE,IMG_SHAPE,3),
    include_top=False)

for layer in base_model.layers:
    layer.trainable = False

inputs = tf.keras.Input(shape=(IMG_SHAPE,IMG_SHAPE,3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(5)(x)
model = tf.keras.Model(inputs, outputs)

print("Compile Model   .compile")
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("Train the Model")
epochs = 20
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_data_gen.n / float(batch_size)))
)
# epochs = 8 loss: 0.8837 - accuracy: 0.6596 - val_loss: 0.7550 - val_accuracy: 0.7252
# 0.735 for batch 66 18 epochs
# 0.743 batch 100 20epochs
# 0.755 batch 45 20epochs
# 0.805 ResNet50V2

print("val_accuracy: ", history.history['val_accuracy'][-1])

print("Plot Training and Validation Graphs")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
