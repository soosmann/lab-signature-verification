import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

"""
DEFINITION OF THE FUNCTIONS FOR THE EXTRACTION OF THE 4 GLOBAL FEATURES
"""


def Ttotal(x: np.ndarray, fs: int) -> float:
    """
    Calculates time in seconds depending on herz.

    Parameters:
        x: The np array with captured values.
        fs: The frequency of captures per second.

    Returns:
        float: Time in seconds for signature.
    """
    time = len(x) / fs
    return time


def Npenups(p: np.ndarray) -> int:
    """
    Number of pen-ups (number of times the pen was lifted). It means the number of times (not the number of samples) that p is equal to 0.

    Parameters:
        p (np.ndarray): List with captured pressure values

    Returns:
        int: Count of penups
    """
    # create list of True for pendown and False for penup
    pen_down = p > 0
    # when pasting this list to np.diff and converting bool to int, 0 and 1 results, np.diff produces list of 0s, 1s and -1s
    transitions = np.diff(pen_down.astype(int))
    # what is wanted is the transitions 1 → 0 for penup
    num_penups = np.sum(transitions == -1)
    return num_penups


def Tpendown(p) -> float:
    """
    Duration of pen-down (signal p is different to 0) Td divided by the total duration T: Td/T.
    Basically the percentage of pendown during signature.

    Parameters:
        p (np.ndarray): List with captured pressure values

    Returns:
        float: Total percentage of pendown during signature.
    """
    total_samples = len(p)
    pen_down_samples = np.sum(p > 0)
    total_pendown = pen_down_samples / total_samples

    return total_pendown


def Ppendown(p):
    """
    Average pressure in pen-down (signal p is different to 0).

    Parameters:
        p (np.ndarray): Pressure captures list.

    Returns:
        val: Average pressure value when writing.
    """
    # filter out the samples that are actual pendowns
    pen_down_values = p[p > 0]
    if len(pen_down_values) == 0:
        return 0
    avg_pendown_pressure = np.mean(pen_down_values)

    return avg_pendown_pressure


def featureExtractor(x, p):  # y
    """
    This function receives as input data (x,y,p) of a given signature and output
    the feature vector FeatVect containing the 4 global features.

    Parameters:
        x: The np array with captured values for x axis.
        p: Array with captured pressure values
    """
    seconds = Ttotal(x, FS)
    number_penups = Npenups(p)
    total_pendown = Tpendown(p)
    avg_pressure_pendown = Ppendown(p)

    return np.array([seconds, number_penups, total_pendown, avg_pressure_pendown])


"""
MAIN

GENERATION OF BiosecurIDparameters MATRIXES FOR GENUINE AND FORGERY
"""

FS = 200
SHOULD_MAKE_PLOT = True

genuine_ids = [1, 2, 6, 7]
forgery_ids = [3, 4, 5]

BiosecurIDparameters_genuine = np.ones((50, 16, 4))
BiosecurIDparameters_forgery = np.ones((50, 12, 4))

for user in range(1, 51):

    g_index = 0
    f_index = 0

    for session in range(1, 5):

        for sig in genuine_ids:
            filename = (
                f"u1{user:03d}s{session:04d}_sg{sig:04d}.mat"  # u1001s0001_sg0001
            )
            data = scipy.io.loadmat(os.path.join("DB", filename))

            x = data["x"].squeeze()
            y = data["y"].squeeze()
            p = data["p"].squeeze()

            features = featureExtractor(x, p)  # y
            BiosecurIDparameters_genuine[user - 1, g_index, :] = features
            g_index += 1

        for sig in forgery_ids:
            filename = f"u1{user:03d}s{session:04d}_sg{sig:04d}.mat"
            data = scipy.io.loadmat(os.path.join("DB", filename))

            x = data["x"].squeeze()
            y = data["y"].squeeze()
            p = data["p"].squeeze()

            features = featureExtractor(x, p)  # y
            BiosecurIDparameters_forgery[user - 1, f_index, :] = features
            f_index += 1

scipy.io.savemat(
    "BiosecurIDparameters_genuine.mat",
    {"BiosecurIDparameters_genuine": BiosecurIDparameters_genuine},
)
scipy.io.savemat(
    "BiosecurIDparameters_forgery.mat",
    {"BiosecurIDparameters_forgery": BiosecurIDparameters_forgery},
)

if SHOULD_MAKE_PLOT:
    feature_names = [
        "Total Duration (T)",
        "Number of Pen-Ups",
        "Pen-Down Ratio (Td/T)",
        "Average Pressure (Pen-Down)",
    ]

    for i in range(4):

        genuine_values = BiosecurIDparameters_genuine[:, :, i].flatten()
        forgery_values = BiosecurIDparameters_forgery[:, :, i].flatten()

        hist_g, bins_g = np.histogram(genuine_values, bins="auto")
        hist_f, bins_f = np.histogram(forgery_values, bins="auto")

        # Normalize between 0 and 1
        hist_g = hist_g / np.sum(hist_g)
        hist_f = hist_f / np.sum(hist_f)

        plt.figure()
        plt.bar(bins_g[:-1], hist_g, width=np.diff(bins_g), alpha=0.5, label="Genuine")
        plt.bar(bins_f[:-1], hist_f, width=np.diff(bins_f), alpha=0.5, label="Forgery")

        plt.title(feature_names[i])
        plt.xlabel("Feature Value")
        plt.ylabel("Normalized Frequency")
        plt.legend()
        plt.show()
