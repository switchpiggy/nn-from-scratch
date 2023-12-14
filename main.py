import neural_network
import numpy as np

def main() -> None:
    Net = neural_network.DenseNetwork([1, 10, 1])
    Net.train(1000, np.arange(0, 0.5, 0.01), np.arange(0, 1, 0.02), 0.001)
    print(Net.predict([0.01, 0.02, 0.03]))

main()