import io
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.layers import TextVectorization

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
train_ds = tf.keras.utils.text_dataset_from_directory(
   'aclImdb/train', batch_size=batch_size, validation_split=0.2,
   subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory(
   'aclImdb/train', batch_size=batch_size, validation_split=0.2,
   subset='validation', seed=seed)

for text_batch, label_batch in train_ds.take(1):
   for i in range(5):
      print(label_batch[i].numpy(), text_batch.numpy()[i])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Embed a 1,000 word vocabulary into 5 dimensions.
embedding_dim = 16
embedding_layer = tf.keras.layers.Embedding(10000, embedding_dim)

result = embedding_layer(tf.constant([1, 2, 3]))
result.numpy()

result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
result.shape


# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
   lowercase = tf.strings.lower(input_data)
   stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
   return tf.strings.regex_replace(stripped_html,
                                   '[%s]' % re.escape(string.punctuation), '')


# Vocabulary size and number of words in a sequence.
vocab_size = 20000
sequence_length = 250
# Use the text vectorization layer to normalize, split, and map strings to
# integers. Note that the layer uses the custom standardization defined above.
# Set maximum_sequence length as all samples are not of the same length.
vectorize_layer = TextVectorization(
   standardize=custom_standardization,
   max_tokens=vocab_size,
   output_mode='int',
   output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

model = Sequential([
   vectorize_layer,
   embedding_layer,
   Conv1D(128, 5, activation='relu', padding='causal'),
   MaxPooling1D(4),
   Conv1D(256, 5, activation='relu', padding='causal'),
   MaxPooling1D(4),
   GlobalAveragePooling1D(),
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

EPOCHS: int = 5
model.fit(
   train_ds,
   validation_data=val_ds,
   epochs=EPOCHS,
   callbacks=[tensorboard_callback])

# 5: loss: 0.2304 - accuracy: 0.9151 - val_loss: 0.3127 - val_accuracy: 0.8766
print("\n")
val_loss, val_acc = model.evaluate(val_ds)

weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
   if index == 0:
      continue  # skip 0, it's padding.
   vec = weights[index]
   out_v.write('\t'.join([str(x) for x in vec]) + "\n")
   out_m.write(word + "\n")
out_v.close()
out_m.close()
