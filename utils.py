import numpy as np


def timestamps_to_beat_vector(beat_ts: np.ndarray, audio_track: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert an array of beat timestamps to a beat vector (vector where 1 == beat and 0 otherwise).

    :param beat_ts: array with timestamps of the individual beats in seconds
    :param audio_track: the audio track as an array
    :param sample_rate: sample rate of the audio track
    :return: array of the same length as the audio track, where 1 indicates a beat
    """
    num_samples = len(audio_track)

    beat_indices = np.round(beat_ts * sample_rate).astype(int)
    beat_vector = np.zeros(num_samples, dtype=int)
    beat_vector[beat_indices] = 1

    return beat_vector


def beat_vector_to_timestamps(beat_vector: np.ndarray, sample_rate: int, threshold: float = 0.4) -> np.ndarray:
    """Convert a beat vector (1 == beat, 0 otherwise) to an array of beat timestamps in seconds.

    :param beat_vector: a 1D array where 1 indicates a beat and 0 otherwise
    :param sample_rate: sample rate of the audio track
    :param threshold: probability threshold for considering the timestamp as a beat
    :return: array with timestamps of the individual beats in seconds
    """
    beat_indices = np.where(beat_vector >= threshold)[0]
    beat_ts = beat_indices / sample_rate
    return beat_ts


def split_array(arr: np.ndarray, segment_length: int) -> list[np.ndarray]:
    """Split the array arr into smaller segments.

    :param arr: array to split
    :param segment_length: length of the segments
    :return: List with the individual segments. The last segment contains the remaining part of the original
        array and might not have the required length.
    """
    return [arr[i : i + segment_length] for i in range(0, len(arr), segment_length)]


def sliding_window_split(arr: np.ndarray, segment_size: int, stride: int):
    """Split an array into segments using a rolling window with the given size and stride.
    The last segment will be excluded if not fully complete.

    :param arr: array to split
    :param segment_size: size of each segment
    :param stride: the stride (step size) between consecutive windows.
    :return: a new array where each element is a segment of the original array
    """
    num_windows = (arr.size - segment_size) // stride + 1  # calculate number of full windows

    if num_windows <= 0:
        return np.empty((0, segment_size))  # return an empty array if no full segment can fit

    windowed = np.lib.stride_tricks.sliding_window_view(arr, segment_size)
    windowed = windowed[::stride]
    return windowed.copy()


def calculate_bpm(timestamps: np.ndarray) -> float:
    """Calculate a tempo in BPM given an array of timestamps with beats.

    :param timestamps: array with timestamps (in second) when beats occur
    :return: the tempo estimated from the timestamps
    """
    intervals = np.diff(timestamps)
    median_interval = np.median(intervals)
    return 60 / median_interval
