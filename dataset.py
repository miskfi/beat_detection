import typing

import mirdata
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchaudio
from tqdm import tqdm

from utils import timestamps_to_beat_vector, sliding_window_split

SAMPLE_RATE = 44100


def download_dataset(path: str = "data"):
    """Download the dataset.

    :param path: path where the data will be stored
    """
    mirdata_dataset = mirdata.initialize("ballroom", data_home=path)
    mirdata_dataset.download()
    mirdata_dataset.validate()


class BallroomDataset(Dataset):
    def __init__(
        self,
        path: str = "data",
        validate: bool = True,
        n_freq_bins: int = 256,
        time_steps: int = 1028,
        segment_len: int = 5,
        stride: int = 1,
    ):
        """Initialize the dataset. The data must already be downloaded (to download the data use the
        'download_dataset' function).

        :param path: path where the downloaded data is stored
        :param validate: should the downloaded data be validated (check if the files aren't corrupted or out of date)
        :param n_freq_bins: desired number of frequency bins of the input spectrogram
        :param time_steps: desired number of time steps of the spectrgram and beat vector
        :param segment_len: length of the audio segments sliced by the sliding window (in seconds)
        :param stride: stride of the sliding window (in seconds)
        """
        super().__init__()

        mirdata_dataset = mirdata.initialize("ballroom", data_home=path)
        if validate:
            print("Validating the dataset")
            mirdata_dataset.validate()

        self.n_freq_bins = n_freq_bins
        self.time_steps = time_steps
        self.spectrogram_transformer = torchaudio.transforms.Spectrogram(
            n_fft=self.n_freq_bins * 2, hop_length=(SAMPLE_RATE * segment_len) // self.time_steps
        )
        # factor with which the segments are downsampled to self.time_steps length
        self.downsample_factor = SAMPLE_RATE * segment_len // self.time_steps

        self.tracks = []

        # Iterate through the individual tracks and split them into 5 second segments. These segments will then
        # be used as the individual samples.
        print("Processing the tracks")
        for track_id in tqdm(mirdata_dataset.track_ids):
            track = mirdata_dataset.track(track_id)
            audio, sample_rate = track.audio
            beat_vector = timestamps_to_beat_vector(track.beats.times, audio, sample_rate)

            # split each track into smaller segments using a sliding window
            audio_segments = sliding_window_split(audio, sample_rate * segment_len, sample_rate * stride)
            beat_vector_segments = sliding_window_split(beat_vector, sample_rate * segment_len, sample_rate * stride)

            for audio_segment, beat_vector_segment in zip(audio_segments, beat_vector_segments):
                # only include segments with more than 3 beats
                if np.count_nonzero(beat_vector_segment) > 3:
                    self.tracks.append(self.process_sample(audio_segment, beat_vector_segment))

            # explicitly free some data to avoid running out of memory
            del audio_segments
            del beat_vector_segments
            del beat_vector

    def __len__(self):
        """Size of the dataset (number of tracks)."""
        return len(self.tracks)

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        """Return the data for a single track.

        :param idx: index of the track
        :return: tuple of (spectrogram, beat_vector).
            The spectrogram has the shape [channels, n_freq_bins, time].
            The beat vector has the shape [time].
        """
        return self.tracks[idx]

    def process_sample(self, audio: np.ndarray, beat_vector: np.ndarray) -> tuple[torch.tensor, torch.tensor]:
        """Process a sample with an audio segment & beat vector segment. The raw audio is converted to a
        spectrogram and the beat vector is downsampled to the same length as the audio spectrogram.

        :param audio: array with the raw audio
        :param beat_vector: array with the original beat vector
        :return: processed audio & beat vector as a tuple of PyTorch tensors
        """
        # convert the audio track to a spectrogram
        spectrogram = self.spectrogram_transformer(torch.from_numpy(audio))
        # add a channels dimension and trim the spectrogram to 1 x n_freq_bins x time_steps
        spectrogram = spectrogram[: self.n_freq_bins, : self.time_steps].unsqueeze(0)

        beat_vector = torch.from_numpy(beat_vector)

        # downsample the beat vector to the same length as the spectrogram
        beat_vector = beat_vector[: self.downsample_factor * self.time_steps]  # trim in case it's not divisible
        beat_vector = beat_vector.view(self.time_steps, self.downsample_factor)
        downsampled_beat_vector, _ = torch.max(beat_vector, dim=1)

        return spectrogram, downsampled_beat_vector.float()

    def get_downsample_factor(self):
        return self.downsample_factor


class BallroomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_p: float = 0.7,
        val_p: float = 0.2,
        batch_size: int = 16,
        num_workers: int = 0,
        dataset: typing.Optional[BallroomDataset] = None,
        **dataset_kwargs
    ):
        """Initialize the PyTorch Lightning DataModule.

        :param train_p: percentage of the dataset to be used as the training set
        :param val_p: percentage of the dataset to be used as the validation set
        :param batch_size: number of samples in the batch
        :param num_workers: number of workers
        :param dataset: the ballroom dataset (in case it was already initialized separately)
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset is None:
            self.dataset = BallroomDataset(**dataset_kwargs)
        else:
            self.dataset = dataset

        train_idx, val_idx, test_idx = self.get_split_idx(train_p, val_p)

        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        self.test_dataset = Subset(self.dataset, test_idx)

    def get_split_idx(self, train_p: float, val_p: float):
        total = len(self.dataset)
        idxs = list(range(total))
        train_size = int(train_p * total)
        val_size = int(val_p * total)

        return idxs[:train_size], idxs[train_size : (train_size + val_size)], idxs[(train_size + val_size) :]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
