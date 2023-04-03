import numpy as np
import tensorflow as tf

# Set up training data
def convert_celsius_fahrenheit(celsius_q, fahrenheit_a):
   for i, c in enumerate(celsius_q):
      print("{} degrees Celsius = {} degress Fahrenhet".format(c, fahrenheit_a[i]))

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

convert_celsius_fahrenheit(celsius_q,fahrenheit_a)

# create a layer
layer_10 = tf.keras.layers.Dense(units=1, input_shape=[1])
print("Layer created")
# assemble layers into model
model = tf.keras.Sequential([layer_10])
print("Layer assembled into model")

# compiling model with loss and optimize functions
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
print("Finished compiling model")
# train model
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training model")

print("Model predicting 100 ºC conversion: ",model.predict([100]), "Fahrenheit Degrees")

print("These are the layer variables: {}".format(layer_10.get_weights()))


print("Start 3 layer model")
layer_11 = tf.keras.layers.Dense(units=4, input_shape=[1])
layer_12 = tf.keras.layers.Dense(units=4)
layer_13 = tf.keras.layers.Dense(units=1)
model2 = tf.keras.Sequential([layer_11, layer_12, layer_13])
model2.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model2.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the 3 layer model")
print("Model2 predicting 100 ºC conversion: ",model2.predict([100]), "Fahrenheit Degrees")
print("These are the layer11 variables: {}".format(layer_11.get_weights()))
print("These are the layer12 variables: {}".format(layer_12.get_weights()))
print("These are the layer13 variables: {}".format(layer_13.get_weights()))
