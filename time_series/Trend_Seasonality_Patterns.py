import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None, label=None):
   plt.plot(time[start:end], series[start:end], format, label=label)
   plt.xlabel("Time")
   plt.ylabel("Value")
   if label:
      plt.legend(fontsize=14)
   plt.grid(True)

print("Trend and Seasonality")
def trend(time, slope=0):
   return slope * time

print("creating time series that trends upwards")
time = np.arange(4 * 365 + 1)
baseline = 10
series = baseline + trend(time, 0.1)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

print(time)
print(series)

print("generate a time series with seasonal pattern")

def seasonal_pattern(season_time):
   """just an arbitrary pattern, you can change it if you wish"""
   return np.where(season_time < 0.4,
                   np.cos(season_time * 2 * np.pi),
                          1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
   """Repeats the same pattern at each period"""
   season_time = ((time + phase) % period) / period
   return amplitude * seasonal_pattern(season_time)

amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

print("Time Series with both trend and seasonality")
slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

print("Now lets define NOISE")
def white_noise(time, noise_level=1, seed=None):
   rnd = np.random.RandomState(seed)
   return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, noise)
plt.show()

series += noise
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


























































































































































