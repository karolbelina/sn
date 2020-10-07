import numpy as np


class SimplePerceptron:
    def __init__(self, size, learning_rate, threshold=None, logic='bipolar'):
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=size)
        self.bias = threshold
        if logic in ['unipolar', 'bipolar']:
            self.logic = logic
        else:
            raise ValueError(f"'{logic}' is not a valid logic type")

    def fit(self, X, y):
        while True:
            error = False
            for x, d in zip(X, y):
                z = np.dot(x, self.weights) + self.bias
                a = self._activation(z)
                delta = d - a
                if delta != 0:
                    error = True
                    self.weights = self.weights + self.learning_rate * delta * x
            if not error:
                return

    def predict(self, X):
        return self._activation(np.dot(X, self.weights) + self.bias)

    def _activation(self, x):
        return 1 if x > 0 else (-1 if self.logic == 'bipolar' else 0)
    
    def __repr__(self):
        return f"SimplePerceptron(size={self.weights.shape[0]}," \
        f"learning_rate={self.learning_rate},threshold={self.bias},logic='{self.logic}')"
