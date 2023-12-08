import neuron, network

def main() -> None:
    Net = network.DenseNetwork(4, 3, [1, 2, 3])
    print(Net.predict([1, 1, 1, 1]))

main()