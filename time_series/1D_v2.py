import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow import keras

keras = tf.keras


def plot_series(time, series, format="-", start=0, end=None, label=None):
   plt.plot(time[start:end], series[start:end], format, label=label)
   plt.xlabel("Time")
   plt.ylabel("Value")
   if label:
      plt.legend(fontsize=14)
   plt.grid(True)


def trend(time, slope=0):
   return slope * time


def seasonal_pattern(season_time):
   """Just an arbitrary pattern, you can change it if you wish"""
   return np.where(season_time < 0.4,
                   np.cos(season_time * 2 * np.pi),
                   1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
   """Repeats the same pattern at each period"""
   season_time = ((time + phase) % period) / period
   return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
   rnd = np.random.RandomState(seed)
   return rnd.randn(len(time)) * noise_level


def seq2seq_window_dataset(series, window_size, batch_size=32,
                           shuffle_buffer=1000):
   series = tf.expand_dims(series, axis=-1)
   ds = tf.data.Dataset.from_tensor_slices(series)
   ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
   ds = ds.flat_map(lambda w: w.batch(window_size + 1))
   ds = ds.shuffle(shuffle_buffer)
   ds = ds.map(lambda w: (w[:-1], w[1:]))
   return ds.batch(batch_size).prefetch(1)


def model_forecast(model, series, window_size):
   ds = tf.data.Dataset.from_tensor_slices(series)
   ds = ds.window(window_size, shift=1, drop_remainder=True)
   ds = ds.flat_map(lambda w: w.batch(window_size))
   ds = ds.batch(32).prefetch(1)
   forecast = model.predict(ds)
   return forecast

time = np.arange(4 * 365 + 1)

slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

series += noise

# plt.figure(figsize=(10, 6))
# plot_series(time, series)
# plt.show()

# Split the data into training and validation sets
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Create the windowed dataset
window_size = 30
train_set = seq2seq_window_dataset(x_train, window_size, batch_size=128)
valid_set = seq2seq_window_dataset(x_valid, window_size, batch_size=128)

# Build and compile the model
model = keras.models.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=5,
                        strides=1, padding="causal",
                        activation="relu",
                        input_shape=[None, 1]),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(1)
])

# Compile the model
optimizer = keras.optimizers.SGD(lr=5e-5, momentum=0.9)
model.compile(loss=keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

EPOCHS = 1500
# Define the ReduceLROnPlateau callback
# Train the model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=60, verbose=1)
history = model.fit(train_set, epochs=EPOCHS, validation_data=valid_set, callbacks=[reduce_lr])

# Plot the learning rate vs. MAE
lrs = 5e-5 * (0.5 ** (np.arange(EPOCHS) / 5))
plt.semilogx(reduce_lr, history.history["loss"])
plt.axis([1e-5, 1e-1, 0, 30])
plt.show() # 2.14e-5 best lr

# Make forecasts
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

# Plot the results
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
plt.show()

# Evaluate the model
mae = keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
print("MAE:", mae)
# Epoch 1500/1500
# 8/8 [==============================] - 0s 19ms/step - loss: 3.9036 - mae: 4.3706 - val_loss: 4.7276 - val_mae: 5.2065 - lr: 2.0480e-05
# 45/45 [==============================] - 1s 6ms/step
# MAE: 4.905869