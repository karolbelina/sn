from abc import ABC, abstractmethod
from activation_fns import ActivationFunction
from random_fns import RandomFunction

import numpy as np


class Layer(ABC):
    @abstractmethod
    def feedforward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backpropagate(self, grad: np.ndarray) -> np.ndarray:
        pass


class FullyConnectedLayer(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        random_fn: RandomFunction,
        activation_fn: ActivationFunction
    ) -> None:
        self._activation_fn = activation_fn
        self._weights = random_fn((output_size, input_size))
        self._biases = random_fn((output_size, 1))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        z = np.matmul(self._weights, x) + self._biases
        a = self._activation_fn(z)

        return a
    
    def backpropagate(self, grad: np.ndarray) -> np.ndarray:
        pass
