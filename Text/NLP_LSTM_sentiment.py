import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')

# Extract out sentences and labels
sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

# Print some example sentences and labels
for x in range(2):
  print(sentences[x])
  print(labels[x])
  print("\n")

print("Create a subwords dataset")
vocab_size = 1000
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
   sentences, vocab_size, max_subword_length=5)

# How big is the vocab size?
print("\nVocab size is ", tokenizer.vocab_size)

# Check that the tokenizer works appropriately
num = 5
print(sentences[num])
encoded = tokenizer.encode(sentences[num])
print(encoded)

# Separately print out each subword, decoded
for i in encoded:
  print(tokenizer.decode([i]))

print("\nReplace sentence data with encoded subwords")
for i, sentence in enumerate(sentences):
  sentences[i] = tokenizer.encode(sentence)
# Check the sentences are appropriately replaced
print(sentences[5])

print("\nFinal pre-processing")
max_length = 50
trunc_type = 'post'
padding_type = 'post'

# Pad all sequences
sequences_padded = pad_sequences(sentences, maxlen=max_length,
                                 padding=padding_type, truncating=trunc_type)

# Separate out the sentences and labels into training and test sets
training_size = int(len(sentences) * 0.8)

training_sequences = sequences_padded[0:training_size]
testing_sequences = sequences_padded[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Make labels into numpy arrays for use with the network later
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

print("\nCreate the model using an Embedding")
embedding_dim = 16

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

print("\nTrain the model")
num_epochs = 30
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(training_sequences, training_labels_final, epochs=num_epochs,
                    validation_data=(testing_sequences, testing_labels_final))

# Epoch 30/30
# 50/50 [==============================] - 0s 5ms/step - loss: 0.1690 - accuracy: 0.9510 - val_loss: 0.5912 - val_accuracy: 0.7569


def plot_graphs(history, string):
   plt.plot(history.history[string])
   plt.plot(history.history['val_' + string])
   plt.xlabel("Epochs")
   plt.ylabel(string)
   plt.legend([string, 'val_' + string])
   plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

print("val_accuracy: ", history.history['val_accuracy'][-1])

print("\nDefine a function to predict the sentiment of reviews")


# Define a function to take a series of reviews
# and predict whether each one is a positive or negative review

# max_length = 100 # previously defined

def predict_review(model, new_sentences, maxlen=max_length, show_padded_sequence=True):
   # Keep the original sentences so that we can keep using them later
   # Create an array to hold the encoded sequences
   new_sequences = []

   # Convert the new reviews to sequences
   for i, frvw in enumerate(new_sentences):
      new_sequences.append(tokenizer.encode(frvw))

   trunc_type = 'post'
   padding_type = 'post'

   # Pad all sequences for the new reviews
   new_reviews_padded = pad_sequences(new_sequences, maxlen=max_length,
                                      padding=padding_type, truncating=trunc_type)

   classes = model.predict(new_reviews_padded)

   # The closer the class is to 1, the more positive the review is
   for x in range(len(new_sentences)):

      # We can see the padded sequence if desired
      # Print the sequence
      if (show_padded_sequence):
         print(new_reviews_padded[x])
      # Print the review as text
      print(new_sentences[x])
      # Print its predicted class
      print(classes[x])
      print("\n")

# Use the model to predict some reviews
fake_reviews = ["I love this phone",
                "Everything was cold",
                "Everything was hot exactly as I wanted",
                "Everything was green",
                "the host seated us immediately",
                "they gave us free chocolate cake",
                "we couldn't hear each other talk because of the shouting in the kitchen"
              ]

predict_review(model, fake_reviews)

print("\nDefine a function to train and show the results of models with different layers")
def fit_model_now (model, sentences) :
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.summary()
  history = model.fit(training_sequences, training_labels_final, epochs=num_epochs,
                      validation_data=(testing_sequences, testing_labels_final))
  return history

def plot_results (history):
  plot_graphs(history, "accuracy")
  plot_graphs(history, "loss")

def fit_model_and_show_results (model, sentences):
  history = fit_model_now(model, sentences)
  plot_results(history)
  predict_review(model, sentences)

print("\nAdd a bidirectional LSTM")
# Define the model
model_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

# Compile and train the model and then show the predictions for our extra sentences
fit_model_and_show_results(model_bidi_lstm, fake_reviews)
# Epoch 30/30
# 50/50 [==============================] - 1s 10ms/step - loss: 0.1783 - accuracy: 0.9642 - val_loss: 1.5663 - val_accuracy: 0.7318

print("\nUse multiple bidirectional layers")
model_multiple_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,
                                                       return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

fit_model_and_show_results(model_multiple_bidi_lstm, fake_reviews)

# Epoch 30/30
# 50/50 [==============================] - 1s 18ms/step - loss: 0.0756 - accuracy: 0.9818 - val_loss: 1.1458 - val_accuracy: 0.7519

print("val_accuracy: ", history.history['val_accuracy'][-1])

print("\nCompare predictions for all the models")
my_reviews =["lovely", "dreadful", "stay away",
             "everything was hot exactly as I wanted",
             "everything was not exactly as I wanted",
             "they gave us free chocolate cake",
             "I've never eaten anything so spicy in my life, my throat burned for hours",
             "for a phone that is as expensive as this one I expect it to be much easier to use than this thing is",
             "we left there very full for a low price so I'd say you just can't go wrong at this place",
             "that place does not have quality meals and it isn't a good place to go for dinner",
             ]

print("===================================\n","Embeddings only:\n", "===================================",)
predict_review(model, my_reviews, show_padded_sequence=False)

print("===================================\n", "With a single bidirectional LSTM:\n", "===================================")
predict_review(model_bidi_lstm, my_reviews, show_padded_sequence=False)

print("===================================\n","With two bidirectional LSTMs:\n", "===================================")
predict_review(model_multiple_bidi_lstm, my_reviews, show_padded_sequence=False)
