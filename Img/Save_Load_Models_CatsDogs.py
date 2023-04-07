import os
import time
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers

print("Part 1: Load the Cats vs. Dogs Dataset")
(train_examples, validation_examples), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

def format_image(image, label):
  # `hub` image modules exepct their data normalized to the [0,1] range.
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
  return  image, label

num_examples = info.splits['train'].num_examples

BATCH_SIZE = 32
IMAGE_RES = 224

train_batches      = train_examples.cache().shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.cache().map(format_image).batch(BATCH_SIZE).prefetch(1)
# .map then .batch to make sure all imgs are same size during batch

print("Part 2: Transfer Learning with TensorFlow Hub")
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))

feature_extractor.trainable = False

print("Attach a classification head")
model = tf.keras.Sequential([
  feature_extractor,
  layers.Dense(2)
])
model.summary()

print("Train the mode")
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

EPOCHS = 1
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

print("Check the predictions")
class_names = np.array(info.features['label'].names)
class_names
image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()
predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]
predicted_class_names
print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)
plt.figure(figsize=(10,9))
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "blue" if predicted_ids[n] == label_batch[n] else "red"
  plt.title(predicted_class_names[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
plt.show()

print("val_accuracy: ", history.history['val_accuracy'][-1])

print("Part 3: Save as Keras .h5 model")
t = time.time()

export_path_keras = "./{}.h5".format(int(t))
print(export_path_keras)

model.save(export_path_keras)
print("Model size: ", os.path.getsize(export_path_keras))
os.system('ls')

print("Part 4: Load the Keras .h5 Model")
reloaded = tf.keras.models.load_model(
  export_path_keras,
  # `custom_objects` tells keras how to load a `hub.KerasLayer`
  custom_objects={'KerasLayer': hub.KerasLayer})
reloaded.summary()

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

(abs(result_batch - reloaded_result_batch)).max()

print("Keep Training")
EPOCHS = 1
history = reloaded.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)

print("Part 5: Export as SavedModel")
t = time.time()

export_path_sm = "./{}.h5".format(int(t))
print(export_path_sm)

tf.saved_model.save(model, export_path_sm)
os.system('ls'.format(export_path_sm))
print("Model size: ", os.path.getsize(export_path_sm))

print("Part 6: Load SavedModel")
reloaded_sm = tf.saved_model.load(export_path_sm)
reload_sm_result_batch = reloaded_sm(image_batch, training=False).numpy()
(abs(result_batch - reload_sm_result_batch)).max()

print("Part 7: Loading the SavedModel as a Keras Model")
t = time.time()

export_path_sm = "./{}.h5".format(int(t))
print(export_path_sm)
tf.saved_model.save(model, export_path_sm)
print("Model size: ", os.path.getsize(export_path_sm))
reload_sm_keras = tf.keras.models.load_model(
  export_path_sm,
  custom_objects={'KerasLayer': hub.KerasLayer})

reload_sm_keras.summary()
result_batch = model.predict(image_batch)
reload_sm_keras_result_batch = reload_sm_keras.predict(image_batch)

(abs(result_batch - reload_sm_keras_result_batch)).max()

print("Part 8: Download your model")
# Zip the contents of export_path_sm directory
os.system('zip -r model.zip {}.h5'.format(export_path_sm))

# List the files and directories in the current directory
os.system('ls')
