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


def seq2seq_window_dataset(series, window_size, batch_size=128,
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

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set = seq2seq_window_dataset(x_train, window_size,
                                   batch_size=128)

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[None, 1]),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=1, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=2, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=4, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=8, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=16, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=32, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=1, kernel_size=1),
    keras.layers.Lambda(lambda x: x * 200)
])


lr_schedule = keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-4 * 10**(epoch / 20))
optimizer = keras.optimizers.Adam(lr=9e-2)
model.compile(loss="mae",
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-5, 1e-1, 0, 20])
# plt.show()

window_size = 46
train_set = seq2seq_window_dataset(x_train, window_size,
                                   batch_size=125)
valid_set = seq2seq_window_dataset(x_valid, window_size,
                                   batch_size=125)

model = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[None, 1]),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=1, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=2, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=4, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=8, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=16, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=32, kernel_size=2, strides=1,
                        dilation_rate=32, padding="causal", activation="relu"),
    keras.layers.Conv1D(filters=1, kernel_size=1),
    keras.layers.Lambda(lambda x: x * 200)
])


optimizer = keras.optimizers.Adam(lr=9e-3)
model.compile(loss="mae",
              optimizer=optimizer,
              metrics=["mae"])
early_stopping = keras.callbacks.EarlyStopping(patience=1500)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    "my_checkpoint.h5", save_best_only=True)

history = model.fit(train_set, epochs=2500,
                    validation_data=valid_set,
                    callbacks=[early_stopping, model_checkpoint])

model = keras.models.load_model("my_checkpoint.h5")

forecast = model_forecast(model, series[..., np.newaxis], window_size)
forecast = forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, forecast)
plt.show()

mae = tf.keras.metrics.mean_absolute_error(x_valid, forecast).numpy()
print("MAE=", mae)
# Epoch 227/500
# 8/8 [==============================] - 0s 16ms/step - loss: 3.4259 - mae: 3.4259 - val_loss: 5.2729 - val_mae: 5.2729
# 45/45 [==============================] - 1s 4ms/step
# MAE= 4.5363116
