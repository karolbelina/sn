from abc import ABC, abstractmethod
from activation_fns import ActivationFunction
from copy import copy, deepcopy
from random_fns import RandomFunction

import numpy as np


class Layer(ABC):
    @abstractmethod
    def feedforward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backpropagate(self, grad: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def update_weights(self, dw: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def update_biases(self, db: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def __deepcopy__(self, memo):
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
        self._weights = random_fn((input_size, output_size)) / output_size ** 0.5
        self._biases = random_fn((1, output_size))
        self._previous_a = None
        self._z = None

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self._weights + self._biases
        a = self._activation_fn(z)

        self._previous_a = x
        self._z = z

        return a

    def backpropagate(self, dC_da_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        da_dz = self._activation_fn.derivative(self._z)
        dz_db = np.ones((1, dC_da_prev.shape[0]))
        dz_dw = self._previous_a
        dz_da = self._weights

        dC_dz = da_dz * dC_da_prev

        dC_db = dz_db @ dC_dz
        dC_dw = dz_dw.T @ dC_dz
        dC_da = dC_dz @ dz_da.T

        return dC_dw, dC_db, dC_da

    def update_weights(self, dw: np.ndarray) -> None:
        self._weights -= dw
    
    def update_biases(self, db: np.ndarray) -> None:
        self._biases -= db
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
