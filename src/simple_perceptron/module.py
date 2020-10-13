from model import Model
from random_fns import RandomFunction
from typing import Optional

import numpy as np


class SimplePerceptron(Model):
    def __init__(
        self,
        input_size: int,
        random_fn: RandomFunction,
        learning_rate: float = 1e-3,
        threshold: Optional[float] = None,
        logic: str = 'bipolar'
    ) -> None:
        self._threshold = threshold if threshold is not None else 0
        if logic == 'unipolar':
            self._low = 0
        elif logic == 'bipolar':
            self._low = -1
        else:
            raise ValueError(f"'{logic}' is not a valid logic type")

        self._weights = random_fn((input_size, 1))
        if threshold is None:
            self._bias = random_fn((1, 1))
        else:
            self._bias = None
        self._learning_rate = learning_rate

    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > self._threshold, 1, self._low)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self._weights
        if self._bias is not None:
            z = z + self._bias
        a = self._activation_fn(z)

        return a

    def training_step(self, batch: tuple[np.ndarray, np.ndarray]):
        x, y_hat = batch
        z = x @ self._weights
        if self._bias is not None:
            z = z + self._bias
        y = self._activation_fn(z)

        self._weights += self._learning_rate * x.T @ (y_hat - y)
        if self._bias is not None:
            self._bias += self._learning_rate * np.ones_like(y).T @ (y_hat - y)

    def validation_step(self, val_batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y_hat = val_batch
        y = self(x)
        error = np.abs(y - y_hat).mean(axis=0)

        return error

    def __str__(self) -> str:
        return "Simple perceptron"
