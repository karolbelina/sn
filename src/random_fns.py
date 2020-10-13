from abc import ABC, abstractmethod

import numpy as np


class RandomFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


class Uniform(RandomFunction):
    def __init__(self, low: float, high: float) -> None:
        self.low = low
        self.high = high

    def __call__(self, size: np.ndarray) -> np.ndarray:
        return np.random.uniform(low=self.low, high=self.high, size=size)
