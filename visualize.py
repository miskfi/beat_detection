import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typing


def plot_spectrogram(spectrogram_db, beat_vector: typing.Optional[torch.tensor] = None, title="Spectrogram"):
    """Plot an audio spectrogram and optionally the beats.

    :param spectrogram_db: spectrogram converted to dB
    :param beat_vector: vector with 1s where the beat is
    :param title: title
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db.numpy(), aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Frequency bins")

    if beat_vector is not None:
        beat_indices = np.where(beat_vector == 1)[0]
        for idx in beat_indices:
            plt.axvline(x=idx, color="red", linestyle="-", linewidth=1)

    plt.show()


def plot_metric(ds: pd.Series, ylabel: str, title: str):
    """Plot chart of a metric over epochs.

    :param ds: data series with the metric scores over epochs
    :param ylabel: label of the y-axis
    :param title: title of the chart
    """
    plt.figure(figsize=(7, 4))

    plt.plot(np.arange(len(ds)), ds, label="Validation Loss", marker="o")
    plt.xticks(np.arange(len(ds)))
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_predictions(ground_truth, predictions: np.ndarray, title: str = ""):
    """Plot the predicted probabilities of the beats & the ground truth beats.

    :param ground_truth: ground truth beat vector
    :param predictions: predicted beat vector
    :param title: title of the chart
    """
    beat_indices = np.where(ground_truth == 1)[0]
    for idx in beat_indices:
        plt.axvline(x=idx, color="red", linestyle="-", linewidth=1)

    plt.plot(predictions)
    plt.xlabel("Time step")
    plt.ylabel("Probability")
    plt.title(title)
    plt.show()
