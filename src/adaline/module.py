from model import Model
from random_fns import RandomFunction

import numpy as np


class Adaline(Model):
    def __init__(
        self,
        input_size: int,
        random_fn: RandomFunction,
        learning_rate: float = 1e-3
    ) -> None:
        self._weights = random_fn((input_size, 1))
        self._bias = random_fn((1, 1))
        self._learning_rate = learning_rate

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self._weights + self._bias
        a = Adaline._activation_fn(z)

        return a

    @staticmethod
    def _activation_fn(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, -1)

    def training_step(self, batch: tuple[np.ndarray, np.ndarray]):
        x, y_hat = batch
        y = x @ self._weights + self._bias

        self._weights += self._learning_rate * x.T @ (y_hat - y)
        self._bias += self._learning_rate * np.ones_like(y).T @ (y_hat - y)

    def validation_step(self, val_batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y_hat = val_batch
        y = self(x)
        error = (((y - y_hat) ** 2) / 2).mean(axis=0)

        return error

    def __str__(self) -> str:
        return "ADALINE"
