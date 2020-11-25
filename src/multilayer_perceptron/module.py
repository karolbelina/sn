from activation_fns import ActivationFunction, Sigmoid
from copy import copy, deepcopy
from model import Model
from typing import Optional
from weight_initializers import WeightInitializer

import numpy as np


class MultilayerPerceptron(Model):
    def __init__(
        self,
        layer_sizes: list[int],
        weight_initializer: WeightInitializer,
        activation_fns: Optional[list[ActivationFunction]] = None,
    ) -> None:
        if activation_fns is None:
            activation_fns = [Sigmoid() for _ in range(len(layer_sizes) - 2)]
        assert len(activation_fns) == len(layer_sizes) - 2

        self._layer_sizes = layer_sizes
        self._activation_fns = activation_fns
        self.θ = self._encode_θ(*MultilayerPerceptron._initialize_θ(layer_sizes, weight_initializer))
    
    @staticmethod
    def _get_shapes(layer_sizes: list[int]):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
        bias_shapes = [(1, s) for s in layer_sizes[1:]]

        return weight_shapes, bias_shapes

    @staticmethod
    def _initialize_θ(layer_sizes: list[int], weight_initializer: WeightInitializer) -> tuple[list[np.ndarray], list[np.ndarray]]:
        weight_shapes, bias_shapes = MultilayerPerceptron._get_shapes(layer_sizes)
        weights = [weight_initializer(fan_in, fan_out) for fan_in, fan_out in weight_shapes]
        biases = [np.zeros(s) for s in bias_shapes]

        return weights, biases
    
    def _encode_θ(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
        return np.hstack(w.flatten() for w in weights + biases)

    def _decode_θ(self, θ: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        weight_shapes, bias_shapes = MultilayerPerceptron._get_shapes(self._layer_sizes)

        layer_count = len(self._layer_sizes) - 1
        arrays = []

        decoded = 0
        for a, b in weight_shapes + bias_shapes:
            arrays.append(np.reshape(θ[decoded:decoded + a * b], (a, b)))
            decoded += a * b
        
        return arrays[:layer_count], arrays[layer_count:]
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        # for numerical stability make the maximum of x to be 0
        stable_x = x - x.max(axis=1)[:, None]
        exp_x = np.exp(stable_x)

        return exp_x / exp_x.sum(axis=1)[:, None]
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        weights, biases = self._decode_θ(self.θ)

        for w, b, f in zip(weights[:-1], biases[:-1], self._activation_fns):
            z = x @ w + b
            x = f(z)

        z = x @ weights[-1] + biases[-1]
        y_hat = MultilayerPerceptron.softmax(z)

        return y_hat

    def dC_dθ(self, x: np.ndarray, y: np.ndarray) -> callable:
        m = x.shape[0]

        def dC_dθ(θ: np.ndarray) -> np.ndarray:
            weights, biases = self._decode_θ(θ)
        
            prev_as = []
            zs = []
            
            current_x = x

            # skip the last layer
            for w, b, f in zip(weights[:-1], biases[:-1], self._activation_fns):
                z = current_x @ w + b
                prev_as.append(current_x)
                zs.append(z)
                current_x = f(z)

            # the softmax layer
            z = current_x @ weights[-1]
            z = z + biases[-1]
            prev_as.append(current_x)
            zs.append(z)
            y_hat = MultilayerPerceptron.softmax(z)

            stable_y_hat = np.clip(y_hat, 1e-12, None)

            # backpropagate
            dC_da = stable_y_hat - y

            dC_dws = []
            dC_dbs = []

            # softmax layer
            dz_db = np.ones((1, dC_da.shape[0]))
            dz_dw = prev_as[-1]
            dz_da = weights[-1]

            dC_db = dz_db @ dC_da
            dC_dw = dz_dw.T @ dC_da
            dC_da = dC_da @ dz_da.T

            dC_dws.append(dC_dw)
            dC_dbs.append(dC_db)

            for w, z, prev_a, f in reversed(list(zip(weights[:-1], zs[:-1], prev_as[:-1], self._activation_fns))):
                da_dz = f.derivative(z)
                dz_db = np.ones((1, dC_da.shape[0]))
                dz_dw = prev_a
                dz_da = w

                dC_dz = da_dz * dC_da

                dC_db = dz_db @ dC_dz
                dC_dw = dz_dw.T @ dC_dz
                dC_da = dC_dz @ dz_da.T

                dC_dws.append(dC_dw)
                dC_dbs.append(dC_db)
            
            dC_dws.reverse()
            dC_dbs.reverse()

            θ = self._encode_θ(dC_dws, dC_dbs)
            
            return θ / m
        
        return dC_dθ
