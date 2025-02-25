import einops
import mir_eval
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from dataset import SAMPLE_RATE
from utils import beat_vector_to_timestamps


class BeatDetectionModel(pl.LightningModule):
    def __init__(
        self,
        n_freq_bins: int,
        downsample_factor: int,
        lstm_hidden_size: int = 128,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.3
    ):
        super(BeatDetectionModel, self).__init__()
        self.learning_rate = learning_rate
        self.lstm_hidden_size = lstm_hidden_size
        self.downsample_factor = downsample_factor

        self.loss = nn.BCELoss()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d((2, 1))  # pool only the frequency dimension

        self.lstm = nn.LSTM(
            input_size=(n_freq_bins // 4) * 64,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout_rate)

        # fully connected layer to map LSTM output to the final beat annotations
        self.fc = nn.Linear(2 * lstm_hidden_size, 1)  # bidirectional LSTM doubles the hidden size

    def forward(self, x, apply_sigmoid: bool = True):
        """Forward pass through the network.

        :param x: tensor of shape [batch_size, channels, n_freq_bins, time_dim]
        :param apply_sigmoid: apply sigmoid activation function at the end of the network
        :return: tensor of shape [batch_size, time_dim]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # shape after pooling: [batch_size, 32, n_freq_bins/2, time_dim]

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # shape after pooling: [batch_size, 64, n_freq_bins/4, time_dim]

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # shape after pooling: [batch_size, 128 (channels), n_freq_bins/8, time_dim]

        x = einops.rearrange(x, "b c fb t -> b t (c fb)")
        x, _ = self.lstm(x)  # shape after LSTM: [batch_size, time_dim, 2 * lstm_hidden_size]
        x = self.dropout(x)

        x = self.fc(x).squeeze(-1)  # shape: [batch_size, time_dim]

        if apply_sigmoid:
            x = F.sigmoid(x)

        return x

    def training_step(self, batch, batch_idx):
        spectrograms, labels = batch
        predictions = self(spectrograms)

        loss = self.loss(predictions, labels)
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        spectrograms, labels = batch
        predictions = self(spectrograms)

        res = self.calculate_metrics(labels, predictions)
        res["val_loss"] = [self.loss(predictions, labels).item()]

        # calculate mean metrics across the batch
        res_mean = {k: np.array(v).mean() for k, v in res.items()}
        self.log_dict(res_mean, on_step=False, on_epoch=True)
        return res_mean

    def test_step(self, batch, batch_idx):
        spectrograms, labels = batch
        predictions = self(spectrograms)

        res = self.calculate_metrics(labels, predictions)
        res["test_loss"] = [self.loss(predictions, labels).item()]

        # calculate mean metrics across the batch
        res_mean = {k: np.array(v).mean() for k, v in res.items()}
        self.log_dict(res_mean, on_step=False, on_epoch=True)
        return res_mean

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def calculate_metrics(self, labels, predictions) -> dict[str, list[float]]:
        """Calculate evaluation metrics (F-measure, Cemgil & continuity scores) for a batch.

        :param labels: ground truth beat vectors
        :param predictions: predicted beat vector
        :return: Dict with the metric names as keys and lists of scores for each individual image. If you want
            the metrics scores for the entire batch, claculate the means of the lists.
        """
        res = defaultdict(list)

        # evaluation metrics (calculated separately for each sample)
        for ground_truth, preds in zip(labels.cpu(), predictions.cpu()):
            ground_truth_ts = beat_vector_to_timestamps(ground_truth, SAMPLE_RATE // self.downsample_factor)
            pred_ts = beat_vector_to_timestamps(preds, SAMPLE_RATE // self.downsample_factor)

            res["F-measure"].append(mir_eval.beat.f_measure(ground_truth_ts, pred_ts))
            res["Cemgil"].append(mir_eval.beat.cemgil(ground_truth_ts, pred_ts)[0])

            for metric, value in zip(
                    ["CMLc", "CMLt", "AMLc", "AMLt"], mir_eval.beat.continuity(ground_truth_ts, pred_ts)
            ):
                res[metric].append(value)

        return res
