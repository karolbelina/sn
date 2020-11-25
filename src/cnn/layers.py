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


class ConvolutionLayer(Layer):
    def __init__(
        self,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
        kernel_size: tuple[int, int],
        maps: int,
        stride: int = 1,
    ) -> None:
        Wx, Wy = input_size
        Vx, Vy = output_size
        Fx, Fy = kernel_size
        S = stride
        Px = (S * (Vx - 1) - Wx + Fx) / 2
        Py = (S * (Vy - 1) - Wy + Fy) / 2
        
        assert Px % 1 == 0 and Py % 1 == 0

        self._W = (Wx, Wy)
        self._V = (Vx, Vy)
        self._F = (Fx, Fy)
        self._P = (int(Px), int(Py))
        self._S = S
        self._feature_map_count = maps

        self._feature_maps = None

    def initialize_θ(self, weight_initializer: WeightInitializer) -> np.ndarray:
        Fx, Fy = self._F
        feature_maps = [weight_initializer(Fx, Fy) for _ in range(self._feature_map_count)]

        return self._encode_θ(feature_maps)

    def _decode_θ(self, θ: np.ndarray) -> list[np.ndarray]:
        Fx, Fy = self._F

        arrays = []

        decoded = 0
        for _ in range(self._feature_map_count):
            arrays.append(np.reshape(θ[decoded:decoded + Fx * Fy], (Fx, Fy)))
            decoded += Fx * Fy
        
        return arrays

    def _encode_θ(self, feature_maps: list[np.ndarray]) -> np.ndarray:
        return np.hstack(m.flatten() for m in feature_maps)

    def get_parameter_count(self) -> int:
        Fx, Fy = self._F

        return Fx * Fy * self._feature_map_count

    def attach(self, θ: np.ndarray) -> None:
        self._feature_maps = self._decode_θ(θ)

    @staticmethod
    def _convolve(x: np.ndarray, kernel: np.ndarray):
        s = (x.shape[0],) + kernel.shape + tuple(np.subtract(x.shape[1:], kernel.shape) + 1)
        strd = np.lib.stride_tricks.as_strided
        subM = strd(x, shape=s, strides=(x.strides[0],) + x.strides[1:] * 2)
        return np.einsum('...ij,...ijkl->...kl', kernel, subM)
    
    @staticmethod
    def _pad(x: np.ndarray, padding: tuple[int, int]):
        Px, Py = padding
        padding = [(0, 0), (Px, Px), (Py, Py)]
        return np.pad(x, padding, mode='constant')

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        Wx, Wy = self._W
        Vx, Vy = self._V
        Px, Py = self._P

        return np.hstack([np.reshape(ConvolutionLayer._pad(
            ConvolutionLayer._convolve(
                np.reshape(x, (-1, Wx, Wy)),
                feature_map
            ),
            (Px, Py)
        ), (-1, Vx * Vy)) for feature_map in self._feature_maps])

    def backpropagate(self, dC_da_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grad = self.initialize_θ(lambda x, y: np.zeros((x, y)))

        return grad, dC_da_prev


class MaxPoolingLayer(Layer):
    def __init__(
        self,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> None:
        Ix, Iy = input_size
        Ox, Oy = output_size
        Wx = Ix / Ox
        Wy = Iy / Oy

        assert Wx % 1 == 0 and Wy % 1 == 0

        self._input_size = input_size
        self._window_size = (int(Wx), int(Wy))

    def initialize_θ(self, weight_initializer: WeightInitializer) -> np.ndarray:
        return np.empty((0,))

    def get_parameter_count(self) -> int:
        return 0

    def attach(self, θ: np.ndarray) -> None:
        pass

    def _max_pool(self, x: np.ndarray):
        N = x.shape[0]
        m, n = x.shape[1:]
        mk = m // 2
        nl = n // 2
        q1 = x[:, :mk * 2, :nl * 2].reshape(N, mk, 2, nl, 2).max(axis=(2, 4))
        q2 = x[:, mk * 2:, :nl * 2].reshape(N, m - mk * 2, nl, 2).max(axis=3)
        q3 = x[:, :mk * 2, nl * 2:].reshape(N, mk, 2, n - nl * 2).max(axis=2)
        q4 = x[:, mk * 2:, nl * 2:].reshape(N, m - mk * 2, n - nl * 2)
        a = np.concatenate((q1, q3), axis=2)
        b = np.concatenate((q2, q4), axis=2)
        return np.concatenate((a, b), axis=1)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        Ix, Iy = self._input_size
        Wx, Wy = self._window_size
        Ox, Oy = Ix // Wx, Iy // Wy
        N = x.shape[1] // (Ix * Iy)

        return np.hstack(np.reshape(self._max_pool(a), (-1, Ox * Oy))
                         for a in np.reshape(x, (N, -1, Ix, Iy)))

    def backpropagate(self, dC_da_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.empty((0,)), dC_da_prev


class ActivationFunction(Layer):
    def __init__(self) -> None:
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
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def _derivative(self, x: np.ndarray) -> np.ndarray:
        sig_x = self(x)

        return sig_x * (1 - sig_x)


class ReLU(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)


class HyperbolicTangent(ActivationFunction):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _derivative(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2
