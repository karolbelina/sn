from abc import ABC, abstractmethod
from copy import copy, deepcopy

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def __copy__(self):
        pass
    
    @abstractmethod
    def __deepcopy__(self, memo):
        pass


class Sigmoid(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig_x = self(x)

        return sig_x * (1 - sig_x)
    
    def __copy__(self):
        return Sigmoid()
        
    def __deepcopy__(self, memo):
        result = copy(self)
        memo[id(self)] = result
        return result


class ReLU(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.max(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)
        
    def __copy__(self):
        return ReLU()
        
    def __deepcopy__(self, memo):
        result = copy(self)
        memo[id(self)] = result
        return result


class HyperbolicTangent(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x) ** 2
    
    def __copy__(self):
        return HyperbolicTangent()
        
    def __deepcopy__(self, memo):
        result = copy(self)
        memo[id(self)] = result
        return result


class Softmax(ActivationFunction):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # for numerical stability make the maximum of x to be 0
        stable_x = x - x.max(axis=1)[:, None]
        exp_x = np.exp(stable_x)

        return exp_x / exp_x.sum(axis=1)[:, None]
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
        
    def __copy__(self):
        return Softmax()
        
    def __deepcopy__(self, memo):
        result = copy(self)
        memo[id(self)] = result
        return result
