from data_loader import DataLoader
from model import Model
from random_fns import RandomFunction
from trainer import Trainer
from typing import Optional

import numpy as np


class Adaline(Model):
    def __init__(
        self,
        input_size: int,
        random_fn: RandomFunction
    ) -> None:
        self._weights = random_fn((input_size, 1))
        self._bias = random_fn((1, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self._weights + self._bias
        a = Adaline._activation_fn(z)

        return a

    @staticmethod
    def _activation_fn(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, -1)

    def __str__(self) -> str:
        return "ADALINE"


class AdalineTrainer(Trainer):
    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
    ) -> None:
        self._learning_rate = learning_rate
        self._train_dataloader = train_dataloader
        if val_dataloader is None:
            self._val_dataloader = train_dataloader
        else:
            self._val_dataloader = val_dataloader
    
    def fit(self, model: Adaline) -> float:
        for data_batch in self._train_dataloader.get_batches():
            self._training_step(model, data_batch)

        val_error = self._validate(model, next(self._val_dataloader.get_batches()))

        return val_error

    def _training_step(self, model: Model, batch: tuple[np.ndarray, np.ndarray]):
        x, y_hat = batch
        y = x @ model._weights + model._bias

        model._weights += self._learning_rate * x.T @ (y_hat - y)
        model._bias += self._learning_rate * np.ones_like(y).T @ (y_hat - y)

    def _validate(self, model: Model, val_batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y_hat = val_batch
        y = model(x)
        error = (((y - y_hat) ** 2) / 2).mean(axis=0)

        return error
