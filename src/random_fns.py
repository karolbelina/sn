from abc import ABC, abstractmethod
from copy import copy, deepcopy

import numpy as np


class RandomFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def __copy__(self):
        pass
    
    @abstractmethod
    def __deepcopy__(self, memo):
        pass


class Uniform(RandomFunction):
    def __init__(self, low: float, high: float) -> None:
        self._low = low
        self._high = high

    def __call__(self, size: np.ndarray) -> np.ndarray:
        return np.random.uniform(low=self._low, high=self._high, size=size)

    def __copy__(self):
        return Uniform(self._low, self._high)
        
    def __deepcopy__(self, memo):
        result = copy(self)
        memo[id(self)] = result
        return result


class Normal(RandomFunction):
    def __init__(self, mean: float = 0, std: float = 1) -> None:
        self._mean = mean
        self._std = std

    def __call__(self, size: np.ndarray) -> np.ndarray:
        return np.random.normal(loc=self._mean, scale=self._std, size=size)

    def __copy__(self):
        return Uniform(self._mean, self._std)
        
    def __deepcopy__(self, memo):
        result = copy(self)
        memo[id(self)] = result
        return result
