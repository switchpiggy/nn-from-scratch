import numpy as np
from typing import List

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class Neuron:
    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.bias = 0
        self.weights = [1] * input_dim
        self.output = 0

    def set_bias(self, b: float) -> None:     
        self.bias = b

    def set_weights(self, w: List[float]) -> None:
        if len(w) != self.input_dim:
            raise Exception("Tried to set wrong length.")
        
        self.weights = w

    def forward(self, inp: List[float]) -> None:
        if len(inp) != self.input_dim:
            raise Exception("Tried to input wrong length.") 

        print(self.weights)
        self.output = sigmoid(np.dot(inp, self.weights) + self.bias)
    
    def get_output(self) -> None:
        return self.output