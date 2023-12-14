import numpy as np
from typing import List

def sigma(x):
    return 1./(1. + np.exp(-x))

def sigma_prime(x):
    return sigma(x) * (1. - sigma(x))

class DenseNetwork:
    def __init__(self, layer_sizes: List[int]) -> None:
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(K, J) for J, K in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(L, 1) for L in layer_sizes[1:]]

    def cost_function_derivative(self, x, y):
        return abs(y - x)
    
    def cost_function(self, x, y):
        return y - x

    def backpropagation(self, x: float, y: float) -> None:
        activations = [np.array(x)]
        zs = []
        activation = x

        del_w = [np.zeros(w.shape) for w in self.weights]
        del_b = [np.zeros(b.shape) for b in self.biases]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = sigma(z)
            activations.append(activation)
            zs.append(z)

        delta = self.cost_function_derivative(activations[-1], y) * sigma_prime(zs[-1])
        # print(delta)

        del_b[-1] = delta
        del_w[-1] = np.dot(delta, activations[-2].transpose())

        for L in range(2, self.num_layers):
            z = zs[-L]
            prime = sigma_prime(z)
            delta = np.dot(self.weights[-L + 1].transpose(), delta) * prime
            del_b[-L] = delta
            del_w[-L] = np.dot(delta, activations[-L-1].transpose())
            # print(delta)
        return (del_w, del_b)

    def train(self, epochs: int, X: List[float], y: List[float], LR: float) -> None:
        E = epochs
        while epochs > 0:
            print("Epoch {}:".format(E - epochs + 1))
            for inp, outp in zip(X, y):
                del_w, del_b = self.backpropagation(inp, outp)
                self.weights = [w - LR * dw for w, dw in zip(self.weights, del_w)]
                self.biases = [b - LR * db for b, db in zip(self.biases, del_b)]
            epochs -= 1

    def predict(self, X: List[float]) -> None:
        preds = []
        for x in X:
            activation = x
            for w, b in zip(self.weights, self.biases):
                activation = sigma(np.dot(w, activation) + b) 
            preds.append(activation)
        
        return preds