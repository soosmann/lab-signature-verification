import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# config values
file_path = "DB/u1001s0001_sg0003.mat"
use_one_plot = True

data = loadmat(file_path)

# Inspect keys if needed
print(data.keys())

# Adjust variable names if required (depends on dataset format)
x = data["x"].squeeze()
y = data["y"].squeeze()
p = data["p"].squeeze()

# print(type(x))

# print(len(x), len(y), len(p))

# sampling freq is 200 Hz
fs = 200
t = np.arange(len(x)) / fs

if use_one_plot:
    plt.figure(figsize=(12, 10))

    # 1. x as function of y (trajectory)
    plt.subplot(2, 2, 1)
    plt.plot(x, y)
    plt.title("Signature trajectory (x vs y)")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.axis("equal")

    # 2. x as function of time
    plt.subplot(2, 2, 2)
    plt.plot(t, x)
    plt.grid()
    plt.title("x vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("x position")

    # 3. y as function of time
    plt.subplot(2, 2, 3)
    plt.plot(t, y)
    plt.grid()
    plt.title("y vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("y position")

    # 4. pressure as function of time
    plt.subplot(2, 2, 4)
    plt.plot(t, p)
    plt.grid()
    plt.title("Pressure vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure")

    plt.tight_layout()
    plt.show()
else:
    plt.figure()
    plt.plot(x, y)
    plt.title("Signature trajectory (x vs y)")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.axis("equal")

    plt.figure()
    plt.plot(t, x)
    plt.grid()
    plt.title("x vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("x position")

    plt.figure()
    plt.grid()
    plt.plot(t, y)
    plt.title("y vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("y position")

    plt.figure()
    plt.grid()
    plt.plot(t, p)
    plt.title("Pressure vs time")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure")

    plt.show()
