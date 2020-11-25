from .layers import Layer
from model import Model
from weight_initializers import WeightInitializer

import numpy as np


class ConvolutionalNeuralNetwork(Model):
    def __init__(
        self,
        layers: list[Layer],
        weight_initializer: WeightInitializer,
    ) -> None:
        self._layers = layers
        self.θ = np.hstack(layer.initialize_θ(weight_initializer) for layer in layers) # NOTE: are we sure we use hstack here?
    
    def _split_θ(self, θ: np.ndarray) -> list[np.ndarray]:
        parameter_counts = [layer.get_parameter_count() for layer in self._layers]
        parameters = []

        decoded = 0
        for count in parameter_counts:
            parameters.append(θ[decoded:decoded + count])
            decoded += count
        
        return parameters
    
    def _merge_θ(self, θ: list[np.ndarray]) -> np.ndarray:
        return np.hstack(θ)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        # for numerical stability make the maximum of x to be 0
        stable_x = x - x.max(axis=1)[:, None]
        exp_x = np.exp(stable_x)

        return exp_x / exp_x.sum(axis=1)[:, None]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        layer_parameters = self._split_θ(self.θ)
        for layer, parameters in zip(self._layers, layer_parameters):
            layer.attach(parameters)
        
        for layer in self._layers:
            x = layer.feedforward(x)
        
        y_hat = ConvolutionalNeuralNetwork.softmax(x)
        
        return y_hat
    
    def dC_dθ(self, x: np.ndarray, y: np.ndarray) -> callable:
        m = x.shape[0]

        def aux(θ: np.ndarray) -> np.ndarray:
            layer_parameters = self._split_θ(self.θ)
            for layer, parameters in zip(self._layers, layer_parameters):
                layer.attach(parameters)
            
            current_x = x

            for layer in self._layers:
                current_x = layer.feedforward(current_x)
            
            y_hat = ConvolutionalNeuralNetwork.softmax(current_x)
            stable_y_hat = np.clip(y_hat, 1e-12, None)

            # backpropagate
            dC_da = stable_y_hat - y

            dC_dθs = []

            for layer in reversed(self._layers):
                dC_dθ, dC_da = layer.backpropagate(dC_da)
                dC_dθs.append(dC_dθ)
            
            dC_dθs.reverse()

            θ = self._merge_θ(dC_dθs)

            return θ / m

        return aux
