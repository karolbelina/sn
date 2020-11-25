from abc import ABC, abstractmethod
from weight_initializers import WeightInitializer

import numpy as np


class Layer(ABC):
    @abstractmethod
    def initialize_θ(self, weight_initializer: WeightInitializer) -> np.ndarray:
        pass

    @abstractmethod
    def get_parameter_count(self) -> int:
        pass

    @abstractmethod
    def attach(self, θ: np.ndarray) -> None:
        pass

    @abstractmethod
    def feedforward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backpropagate(self, dC_da_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass


class FullyConnectedLayer(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
    ) -> None:
        self._input_size = input_size
        self._output_size = output_size

        self._weights = None
        self._biases = None
        self._previous_a = None
    
    def initialize_θ(self, weight_initializer: WeightInitializer) -> np.ndarray:
        weights = weight_initializer(self._input_size, self._output_size)
        biases = np.zeros((1, self._output_size))

        return self._encode_θ(weights, biases)

    def get_parameter_count(self) -> int:
        weights_count = self._input_size * self._output_size
        biases_count = self._output_size

        return weights_count + biases_count

    def attach(self, θ: np.ndarray) -> None:
        self._weights, self._biases = self._decode_θ(θ)
    
    def _decode_θ(self, θ: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        weights_shape = (self._input_size, self._output_size)
        biases_shape = (1, self._output_size)

        halfpoint = self._input_size * self._output_size

        weights = np.reshape(θ[:halfpoint], weights_shape)
        biases = np.reshape(θ[halfpoint:], biases_shape)

        return weights, biases
    
    def _encode_θ(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
        return np.hstack([weights.flatten(), biases.flatten()])

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self._weights + self._biases

        self._previous_a = x

        return z

    def backpropagate(self, dC_da_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dz_db = np.ones((1, dC_da_prev.shape[0]))
        dz_dw = self._previous_a
        dz_da = self._weights

        dC_db = dz_db @ dC_da_prev
        dC_dw = dz_dw.T @ dC_da_prev
        dC_da = dC_da_prev @ dz_da.T

        dC_dθ = self._encode_θ(dC_dw, dC_db)

        return dC_dθ, dC_da


class ActivationFunction(Layer):
    def __init__(self, size: int) -> None:
        self._size = size

        self._z = None

    def initialize_θ(self, weight_initializer: WeightInitializer) -> np.ndarray:
        return np.empty((0,))

    def get_parameter_count(self) -> int:
        return 0

    def attach(self, θ: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def _derivative(self, x: np.ndarray) -> np.ndarray:
        pass

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        self._z = x

        return self(x)

    def backpropagate(self, dC_da_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        da_dz = self._derivative(self._z)

        dC_da = da_dz * dC_da_prev

        return np.empty((0,)), dC_da


class Sigmoid(ActivationFunction):
    def __init__(self, size: int) -> None:
        super().__init__(size)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def _derivative(self, x: np.ndarray) -> np.ndarray:
        sig_x = self(x)

        return sig_x * (1 - sig_x)
