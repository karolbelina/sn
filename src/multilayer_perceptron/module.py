from activation_fns import Sigmoid, Softmax
from layers import FullyConnectedLayer as FCL
from model import Model
from random_fns import RandomFunction, Normal
from typing import Optional

import numpy as np


class MultilayerPerceptron(Model):
    def __init__(
        self,
        layer_sizes: list[int],
        activation_fns: Optional[list[callable]] = None,
        random_fn: RandomFunction = Normal(),
        learning_rate: float = 1e-3,
    ) -> None:
        if activation_fns is None:
            activation_fns = [Sigmoid() for _ in range(len(layer_sizes) - 2)]
        if len(activation_fns) == len(layer_sizes) - 2:
            activation_fns.append(Softmax())
        else:
            raise ValueError(f"activation_fns has an incorrect length")

        self._layers = [FCL(a, b, random_fn, activation_fn)
                        for a, b, activation_fn in zip(layer_sizes[:-1], layer_sizes[1:], activation_fns)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            x = layer.feedforward(x)

        return x

    def training_step(self, batch: tuple[np.ndarray, np.ndarray]):
        pass

    def validation_step(self, val_batch: tuple[np.ndarray, np.ndarray]) -> float:
        pass

    def __str__(self) -> str:
        return "Multilayer perceptron"
