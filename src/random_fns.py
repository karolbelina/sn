from abc import ABC, abstractmethod

import numpy as np


class RandomFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


class Uniform(RandomFunction):
    def __init__(self, low: float, high: float) -> None:
        self._low = low
        self._high = high

    def __call__(self, size: np.ndarray) -> np.ndarray:
        return np.random.uniform(low=self._low, high=self._high, size=size)


class Normal(RandomFunction):
    def __init__(self, mean: float = 0, std: float = 1) -> None:
        self._mean = mean
        self._std = std

    def __call__(self, size: np.ndarray) -> np.ndarray:
        return np.random.normal(loc=self._mean, scale=self._std, size=size)
