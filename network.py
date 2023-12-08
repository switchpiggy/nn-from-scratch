import neuron
from typing import List

class DenseLayer:
    def __init__(self, size: int, neurons: List[neuron.Neuron]) -> None:
        self.size = size
        if neurons == None:
            self.neurons = [neuron.Neuron(1)] * size
        else:
            self.neurons = neurons

    def forward(self, inputs: List[float]) -> None:
        for index, _ in enumerate(self.neurons):
            self.neurons[index].forward(inputs)

    def get_outputs(self) -> List[float]: 
        return [N.get_output() for N in self.neurons]
    
class InputLayer:
    def __init__(self, input_dim: List[float]) -> None:
        self.size = input_dim
        self.inputs = [0] * input_dim

    def forward(self, inputs: List[float]) -> None:
        if len(inputs) != self.size:
            raise Exception("Wrong input length!")
        self.inputs = inputs

    def get_outputs(self) -> None:
        return self.inputs

class DenseNetwork:
    def __init__(self, input_dim, num_layers, layer_sizes: List[int]) -> None:
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.layers = [InputLayer(input_dim)]
        for index, L in enumerate(layer_sizes):
            self.layers.append(DenseLayer(L, [neuron.Neuron(self.layers[index].size)] * L))

    def predict(self, inp: float) -> List[float]:
        output = inp
        for L in self.layers:
            L.forward(output)
            output = L.get_outputs()
        
        return output