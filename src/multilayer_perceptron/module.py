from .parameters import Parameters
from activation_fns import ActivationFunction, Sigmoid
from copy import copy, deepcopy
from model import Model
from random_fns import RandomFunction, Normal
from typing import Optional

import numpy as np


def initialize_θ(layer_sizes: list[int], random_fn: RandomFunction) -> Parameters:
    weight_shapes = [(a, b) for a, b in zip(layer_sizes[:-1], layer_sizes[1:])]
    weights = [random_fn(s) for s in weight_shapes]
    biases = [np.zeros((1, s)) for s in layer_sizes[1:]]

    return Parameters(weights, biases)


def softmax(x: np.ndarray) -> np.ndarray:
    # for numerical stability make the maximum of x to be 0
    stable_x = x - x.max(axis=1)[:, None]
    exp_x = np.exp(stable_x)

    return exp_x / exp_x.sum(axis=1)[:, None]


class MultilayerPerceptron(Model):
    def __init__(
        self,
        layer_sizes: list[int],
        activation_fns: Optional[list[ActivationFunction]] = None,
        random_fn: RandomFunction = Normal(),
    ) -> None:
        if activation_fns is None:
            activation_fns = [Sigmoid() for _ in range(len(layer_sizes) - 2)]
        assert len(activation_fns) == len(layer_sizes) - 2

        self._activation_fns = activation_fns
        self._θ = initialize_θ(layer_sizes, random_fn)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        for w, b, f in zip(self._θ.weights[:-1], self._θ.biases[:-1], self._activation_fns):
            z = x @ w + b
            x = f(z)

        z = x @ self._θ.weights[-1] + self._θ.biases[-1]
        y_hat = softmax(z)

        return y_hat

    def generate_dC_dθ(self, x: np.ndarray, y: np.ndarray) -> callable:
        m = x.shape[0]

        def dC_dθ(θ: Parameters) -> Parameters:
            prev_as = []
            zs = []
            
            current_x = x

            # skip the last layer
            for w, b, f in zip(θ.weights[:-1], θ.biases[:-1], self._activation_fns):
                z = current_x @ w + b
                prev_as.append(current_x)
                zs.append(z)
                current_x = f(z)

            # the softmax layer
            z = current_x @ θ.weights[-1]
            z = z + θ.biases[-1]
            prev_as.append(current_x)
            zs.append(z)
            y_hat = softmax(z)

            stable_y_hat = np.clip(y_hat, 1e-12, None)

            # backpropagate
            dC_da = stable_y_hat - y

            dC_dws = []
            dC_dbs = []

            # softmax layer
            dz_db = np.ones((1, dC_da.shape[0]))
            dz_dw = prev_as[-1]
            dz_da = θ.weights[-1]

            dC_db = dz_db @ dC_da
            dC_dw = dz_dw.T @ dC_da
            dC_da = dC_da @ dz_da.T

            dC_dws.append(dC_dw)
            dC_dbs.append(dC_db)

            for w, z, prev_a, f in reversed(list(zip(θ.weights[:-1], zs, prev_as, self._activation_fns))):
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

            θ = Parameters(dC_dws, dC_dbs)
            
            return θ / m
        
        return dC_dθ
