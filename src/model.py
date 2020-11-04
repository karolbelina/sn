from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        pass
