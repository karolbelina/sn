from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig_x = self(x)

        return sig_x * (1 - sig_x)


class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.max(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)


class HyperbolicTangent(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2


class Softmax(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # for numerical stability make the maximum of x to be 0
        e_x = np.exp(x - np.max(x))

        return e_x / e_x.sum()
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        s_x = self(x)

        return -np.outer(s_x, s_x) + np.diag(s_x.flatten())
