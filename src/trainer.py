from abc import ABC, abstractmethod
from data_loader import DataLoader
from itertools import chain
from model import Model
from typing import Optional

import numpy as np


class Trainer:
    def __init__(self, max_epochs: Optional[int] = None, epsilon: float = 0) -> None:
        self._max_epochs = max_epochs if max_epochs is not None else np.inf
        self._epsilon = epsilon

    def fit(self, model: Model, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        val_error = None
        epoch_number = 0

        while epoch_number < self._max_epochs and (val_error is None or val_error > self._epsilon):
            epoch_number += 1

            for data_batch in train_dataloader.get_batches():
                model.training_step(data_batch)

            val_error = model.validation_step(
                next(val_dataloader.get_batches()))

        return epoch_number
