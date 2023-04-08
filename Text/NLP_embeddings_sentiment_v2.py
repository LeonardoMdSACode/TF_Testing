import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import csv
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


dataset = pd.read_csv('data.csv')

sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()
NUM_WORDS = 10000
EMBEDDING_DIM = 100
MAXLEN = 100
PADDING = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SPLIT = .96

def remove_stopwords(sentence):
   # List of stopwords
   stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",
                "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
                "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll",
                "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
                "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
                "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
                "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while",
                "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
                "your", "yours", "yourself", "yourselves"]

   # Sentence converted to lowercase-only
   sentence = sentence.lower()

   words = sentence.split()
   no_words = [w for w in words if w not in stopwords]
   sentence = " ".join(no_words)

   return sentence


def parse_data_from_file(filename):
   sentences = []
   labels = []
   with open(filename, 'r') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      next(reader)
      for row in reader:
         labels.append(row[2])
         sentence = row[1]
         sentence = remove_stopwords(sentence)
         sentences.append(sentence)

   return sentences, labels
sentences, labels = parse_data_from_file("data.csv")

# GRADED FUNCTIONS: train_val_split
def train_val_split(sentences, labels, training_split):
   ### START CODE HERE

   # Compute the number of sentences that will be used for training (should be an integer)
   train_size = int(len(sentences) * training_split)

   # Split the sentences and labels into train/validation splits
   train_sentences = sentences[:train_size]
   train_labels = labels[:train_size]

   validation_sentences = sentences[train_size:]
   validation_labels = labels[train_size:]

   ### END CODE HERE

   return train_sentences, validation_sentences, train_labels, validation_labels
train_sentences, val_sentences, train_labels, val_labels = train_val_split(sentences, labels, TRAINING_SPLIT)

# GRADED FUNCTION: fit_tokenizer
def fit_tokenizer(train_sentences, num_words, oov_token):
   ### START CODE HERE

   # Instantiate the Tokenizer class, passing in the correct values for num_words and oov_token
   tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

   # Fit the tokenizer to the training sentences
   tokenizer.fit_on_texts(train_sentences)

   ### END CODE HERE

   return tokenizer
tokenizer = fit_tokenizer(train_sentences, NUM_WORDS, OOV_TOKEN)
word_index = tokenizer.word_index

# GRADED FUNCTION: seq_and_pad
def seq_and_pad(sentences, tokenizer, padding, maxlen):
   ### START CODE HERE

   # Convert sentences to sequences
   sequences = tokenizer.texts_to_sequences(sentences)

   # Pad the sequences using the correct padding and maxlen
   padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding=padding)

   ### END CODE HERE

   return padded_sequences
train_padded_seq = seq_and_pad(train_sentences, tokenizer, PADDING, MAXLEN)
val_padded_seq = seq_and_pad(val_sentences, tokenizer, PADDING, MAXLEN)

# GRADED FUNCTION: tokenize_labels
def tokenize_labels(all_labels, split_labels):
   ### START CODE HERE

   # Instantiate the Tokenizer (no additional arguments needed)
   label_tokenizer = Tokenizer()

   # Fit the tokenizer on all the labels
   label_tokenizer.fit_on_texts(all_labels)

   # Convert labels to sequences
   label_seq = label_tokenizer.texts_to_sequences(split_labels)

   # Convert sequences to a numpy array. Don't forget to substact 1 from every entry in the array!
   label_seq_np = np.array(label_seq) - 1

   ### END CODE HERE

   return label_seq_np
train_label_seq = tokenize_labels(labels, train_labels)
val_label_seq = tokenize_labels(labels, val_labels)

# GRADED FUNCTION: create_model
def create_model(num_words, embedding_dim, maxlen):
   tf.random.set_seed(123)

   ### START CODE HERE

   model = tf.keras.Sequential([
      tf.keras.layers.Embedding(num_words, embedding_dim, input_length=maxlen),
      Conv1D(128, 5, activation='relu'),
      GlobalMaxPooling1D(),
      Dense(64, activation='relu'),
      Dropout(0.5),
      Dense(1, activation='sigmoid')
   ])

   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   loss, accuracy = model.evaluate(val_padded_seq, val_label_seq, verbose=False)
   print(f'Validation accuracy: {accuracy:.4f}')
   model.summary()

   ### END CODE HERE

   return model

model = create_model(NUM_WORDS, EMBEDDING_DIM, MAXLEN)

history = model.fit(train_padded_seq, train_label_seq, epochs=4, validation_data=(val_padded_seq, val_label_seq))

# Epoch 4/4
# 60/60 [==============================] - 0s 5ms/step - loss: 0.0465 - accuracy: 0.9864 - val_loss: 0.2067 - val_accuracy: 0.9375

def plot_graphs(history, metric):
   plt.plot(history.history[metric])
   plt.plot(history.history[f'val_{metric}'])
   plt.xlabel("Epochs")
   plt.ylabel(metric)
   plt.legend([metric, f'val_{metric}'])
   plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
