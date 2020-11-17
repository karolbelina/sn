from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def generate_dC_dθ(self, x: np.ndarray, y: np.ndarray) -> callable:
        pass
