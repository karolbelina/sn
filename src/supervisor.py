from data_loader import DataLoader
from model import Model
from trainer import Trainer
from typing import Optional

import numpy as np


class Supervisor:
    def __init__(self, max_epochs: Optional[int] = None, epsilon: float = 0) -> None:
        self._max_epochs = max_epochs if max_epochs is not None else np.inf
        self._epsilon = epsilon

    def run(
        self,
        model: Model,
        trainer: Trainer
    ) -> None:
        val_error = None
        epoch_number = 0

        while epoch_number < self._max_epochs and (val_error is None or val_error > self._epsilon):
            epoch_number += 1

            val_error = trainer.fit(model)

        return epoch_number
