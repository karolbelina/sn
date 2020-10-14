from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def training_step(self, batch: tuple[np.ndarray, np.ndarray]):
        pass

    @abstractmethod
    def validation_step(self, val_batch: tuple[np.ndarray, np.ndarray]):
        pass
