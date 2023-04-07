# Comparing multiple models for sentiment review
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Get the dataset.
# It has 70000 items, so might take a while to download
dataset, info = tfds.load('glue/sst2', with_info=True)
print(info.features)
print(info.features["label"].num_classes)
print(info.features["label"].names)

# Get the training and validation datasets
dataset_train, dataset_validation = dataset['train'], dataset['validation']
print(dataset_train)

# Print some of the entries
for example in dataset_train.take(2):
  review, label = example["sentence"], example["label"]
  print("Review:", review)
  print("Label: %d \n" % label.numpy())

# Get the sentences and the labels
# for both the training and the validation sets
training_reviews = []
training_labels = []

validation_reviews = []
validation_labels = []

# The dataset has 67,000 training entries, but that's a lot to process here!

# If you want to take the entire dataset: WARNING: takes longer!!
# for item in dataset_train.take(-1):

# Take 10,000 reviews
for item in dataset_train.take(10000):
   review, label = item["sentence"], item["label"]
   training_reviews.append(str(review.numpy()))
   training_labels.append(label.numpy())

print("\nNumber of training reviews is: ", len(training_reviews))

# print some of the reviews and labels
for i in range(0, 2):
   print(training_reviews[i])
   print(training_labels[i])

# Get the validation data
# there's only about 800 items, so take them all
for item in dataset_validation.take(-1):
   review, label = item["sentence"], item["label"]
   validation_reviews.append(str(review.numpy()))
   validation_labels.append(label.numpy())

print("\nNumber of validation reviews is: ", len(validation_reviews))

# Print some of the validation reviews and labels
for i in range(0, 2):
   print(validation_reviews[i])
   print(validation_labels[i])

print("Tokenize the words and sequence the sentences")
# There's a total of 21224 words in the reviews
# but many of them are irrelevant like with, it, of, on.
# If we take a subset of the training data, then the vocab
# will be smaller.

# A reasonable review might have about 50 words or so,
# so we can set max_length to 50 (but feel free to change it as you like)

vocab_size = 4000
embedding_dim = 16
max_length = 50
trunc_type = 'post'
pad_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_reviews)
word_index = tokenizer.word_index

print("\nPad the sequences")
# Pad the sequences so that they are all the same length
training_sequences = tokenizer.texts_to_sequences(training_reviews)
training_padded = pad_sequences(training_sequences,maxlen=max_length,
                                truncating=trunc_type, padding=pad_type)

validation_sequences = tokenizer.texts_to_sequences(validation_reviews)
validation_padded = pad_sequences(validation_sequences,maxlen=max_length)

training_labels_final = np.array(training_labels)
validation_labels_final = np.array(validation_labels)


print("\nPlot the accuracy and loss")
def plot_graphs(history, string):
   plt.plot(history.history[string])
   plt.plot(history.history['val_' + string])
   plt.xlabel("Epochs")
   plt.ylabel(string)
   plt.legend([string, 'val_' + string])
   plt.show()

print("\nWrite a function to predict the sentiment of reviews")
# Write some new reviews
review1 = """I loved this movie"""
review2 = """that was the worst movie I've ever seen"""
review3 = """too much violence even for a Bond film"""
review4 = """a captivating recounting of a cherished myth"""
new_reviews = [review1, review2, review3, review4]
# Define a function to prepare the new reviews for use with a model
# and then use the model to predict the sentiment of the new reviews
def predict_review(model, reviews):
  # Create the sequences
  padding_type='post'
  sample_sequences = tokenizer.texts_to_sequences(reviews)
  reviews_padded = pad_sequences(sample_sequences, padding=padding_type,
                                 maxlen=max_length)
  classes = model.predict(reviews_padded)
  for x in range(len(reviews_padded)):
    print(reviews[x])
    print(classes[x])
    print('\n')


print("\nDefine a function to train and show the results of models with different layers")
def fit_model_and_show_results (model, reviews):
   model.compile(loss='binary_crossentropy',
           optimizer=tf.keras.optimizers.Adam(learning_rate),
           metrics=['accuracy'])
   model.summary()
   history = model.fit(training_padded, training_labels_final, epochs=num_epochs,
                      validation_data=(validation_padded, validation_labels_final))
   plot_graphs(history, "accuracy")
   plot_graphs(history, "loss")
   predict_review(model, reviews)
   print("val_accuracy: ", history.history['val_accuracy'][-1])


print("\n Model with Embedding")
num_epochs = 20
# loss: 0.3088 - accuracy: 0.8697 - val_loss: 0.5204 - val_accuracy: 0.7638
model_embedding = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.001

fit_model_and_show_results(model_embedding, new_reviews)

print("\n Use a CNN")
num_epochs = 30
# loss: 0.2397 - accuracy: 0.8957 - val_loss: 0.5576 - val_accuracy: 0.7569
model_cnn = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(16, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Default learning rate for the Adam optimizer is 0.001
# Let's slow down the learning rate by 10.
learning_rate = 0.0001

fit_model_and_show_results(model_cnn, new_reviews)

print("\nUse a GRU")
num_epochs = 30
# loss: 0.3550 - accuracy: 0.8462 - val_loss: 0.5886 - val_accuracy: 0.7729
model_gru = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.00003 # slower than the default learning rate

fit_model_and_show_results(model_gru, new_reviews)

print("\nAdd a bidirectional LSTM")
num_epochs = 30
# loss: 0.4655 - accuracy: 0.8057 - val_loss: 0.7013 - val_accuracy: 0.4450
model_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.00003

fit_model_and_show_results(model_bidi_lstm, new_reviews)

print("\nUse multiple bidirectional LSTMs = RNN")
num_epochs = 30
# loss: 0.1329 - accuracy: 0.9338 - val_loss: 1.6815 - val_accuracy: 0.4083
model_multiple_bidi_lstm = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.0003

fit_model_and_show_results(model_multiple_bidi_lstm, new_reviews)

print("\nTry some more reviews")
# Write some new reviews

review1 = """I loved this movie"""

review2 = """that was the worst movie I've ever seen"""

review3 = """too much violence even for a Bond film"""

review4 = """a captivating recounting of a cherished myth"""

review5 = """I saw this movie yesterday and I was feeling low to start with,
 but it was such a wonderful movie that it lifted my spirits and brightened 
 my day, you can\'t go wrong with a movie with Whoopi Goldberg in it."""

review6 = """I don\'t understand why it received an oscar recommendation
 for best movie, it was long and boring"""

review7 = """the scenery was magnificent, the CGI of the dogs was so realistic I
 thought they were played by real dogs even though they talked!"""

review8 = """The ending was so sad and yet so uplifting at the same time. 
 I'm looking for an excuse to see it again"""

review9 = """I had expected so much more from a movie made by the director 
 who made my most favorite movie ever, I was very disappointed in the tedious 
 story"""

review10 = "I wish I could watch this movie every day for the rest of my life"

more_reviews = [review1, review2, review3, review4, review5, review6, review7,
               review8, review9, review10]
print("============================\n","Embeddings only:\n", "============================")
predict_review(model_embedding, more_reviews)
print("============================\n","With CNN\n", "============================")
predict_review(model_cnn, more_reviews)
print("===========================\n","With bidirectional GRU\n", "============================")
predict_review(model_gru, more_reviews)
print("===========================\n", "With a single bidirectional LSTM:\n", "===========================")
predict_review(model_bidi_lstm, more_reviews)
print("===========================\n", "With multiple bidirectional LSTM:\n", "==========================")
predict_review(model_multiple_bidi_lstm, more_reviews)
