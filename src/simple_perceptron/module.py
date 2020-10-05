import numpy as np


class SimplePerceptron:
    def __init__(self, size, learning_rate, threshold=None, logic='bipolar'):
        self.learning_rate = learning_rate
        self.weights = np.random.standard_normal(size)
        self.bias = threshold
        if logic in ['unipolar', 'bipolar']:
            self.logic = logic
        else:
            raise ValueError(f"'{logic}' is not a valid logic type")

    def predict(self, a):
        return self.activation(np.dot(a, self.weights) + self.bias)

    def train(self, D):
        for x, d in D:
            z = np.dot(x, self.weights) + self.bias
            a = self.activation(z)
            delta = d - a
            self.weights += self.learning_rate * delta * self.weights 

    def activation(self, x):
        return 1 if x > 0 else (-1 if self.logic == 'bipolar' else 0)
    
    def __repr__(self):
        return f"SimplePerceptron(size={self.weights.shape[0]}," \
        f"learning_rate={self.learning_rate},threshold={self.bias},logic='{self.logic}')"
