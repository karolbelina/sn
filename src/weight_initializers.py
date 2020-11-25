from abc import ABC, abstractmethod
from copy import copy, deepcopy

import numpy as np


class WeightInitializer(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass


class Normal(WeightInitializer):
    def __call__(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.randn(fan_in, fan_out)


class Xavier(WeightInitializer):
    def __call__(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.randn(fan_in, fan_out) * np.sqrt(2 / (fan_out + fan_in))


class He(WeightInitializer):
    def __call__(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)
