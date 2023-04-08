import io
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_hub as hub


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
   'aclImdb/train', batch_size=batch_size, validation_split=0.2,
   subset='training', seed=seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
   'aclImdb/train', batch_size=batch_size, validation_split=0.2,
   subset='validation', seed=seed)

for text_batch, label_batch in train_ds.take(1):
   for i in range(5):
      print(label_batch[i].numpy(), text_batch.numpy()[i])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


embed = hub.load("https://tfhub.dev/google/Wiki-words-250-with-normalization/2")

model = Sequential([
    hub.KerasLayer(embed, input_shape=[], dtype=tf.string, trainable=True),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

EPOCHS: int = 10

model.fit(train_ds, validation_data=val_ds,
          epochs=EPOCHS, callbacks=[tensorboard_callback])

# 10: loss: 0.0865 - accuracy: 0.9715 - val_loss: 0.2715 - val_accuracy: 0.9012
print("\n")
val_loss, val_acc = model.evaluate(val_ds)

weights = model.get_layer('keras_layer').get_weights()[0]
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')
for index, word in enumerate(weights):
    if index == 0:
        continue  # skip 0, it's padding.
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(str(index) + "\n")
out_v.close()
out_m.close()
