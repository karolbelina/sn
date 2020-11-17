import numpy as np


class Parameters:
    def __init__(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        assert len(weights) == len(biases)
        for w, b in zip(weights, biases):
            assert w.shape[1] == b.shape[1]
        for a, b in zip(weights[:-1], weights[1:]):
            assert a.shape[1] == b.shape[0]
        for b in biases:
            assert b.shape[0] == 1
        self.weights = weights
        self.biases = biases
    
    def __repr__(self) -> str:
        return f"Parameters(weights={self.weights}, biases={self.biases})"
    

def _impl(name, f: callable):
    def operation(self, other):
        if isinstance(other, int) or isinstance(other, float):
            weights = [f(w, other) for w in self.weights]
            biases = [f(b, other) for b in self.biases]
        elif isinstance(other, Parameters):
            assert len(self.weights) == len(other.weights)
            assert len(self.biases) == len(other.biases)
            weights = [f(a, b) for a, b in zip(self.weights, other.weights)]
            biases = [f(a, b) for a, b in zip(self.biases, other.biases)]
        else:
            raise TypeError()
        return Parameters(weights, biases)
    
    setattr(Parameters, name, operation)


_impl('__add__', lambda a, b: a + b)
_impl('__sub__', lambda a, b: a - b)
_impl('__mul__', lambda a, b: a * b)
_impl('__truediv__', lambda a, b: a / b)
_impl('__radd__', lambda a, b: b + a)
_impl('__rsub__', lambda a, b: b - a)
_impl('__rmul__', lambda a, b: b * a)
_impl('__rtruediv__', lambda a, b: b / a)


def zeros_like(other: Parameters) -> Parameters:
    weights = [np.zeros_like(w) for w in other.weights]
    biases = [np.zeros_like(b) for b in other.biases]

    return Parameters(weights, biases)
