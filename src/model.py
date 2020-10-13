import numpy as np


class Model:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def training_step(self, batch: tuple[np.ndarray, np.ndarray]):
        pass

    def validation_step(self, val_batch: tuple[np.ndarray, np.ndarray]):
        pass
