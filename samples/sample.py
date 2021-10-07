
from nnpylib.networks import NeuralNetwork
from nnpylib.storage import dump_nn, load_nn


if __name__ == "__main__":
    # Training parameters
    inputs = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
        ]
    outputs = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
        ]
    # Set up the network
    hidden_nodes = [4,]
    nn = NeuralNetwork(len(inputs[0]), len(outputs[0]), hidden_nodes,
                       learning_rate=0.1, method="leaky_relu")
    nn.train(inputs, outputs, 10000)
    # Now the network is fit, lets try some predictions
    for i in range(len(inputs)):
        predicted = nn.predict(inputs[i])
        print("input: ", inputs[i],
              "output:", outputs[i],
              "predicted: ", predicted)
    # Dump the network, then load from memory
    data = dump_nn(nn)
    nn2 = load_nn(data)
    # Now, make some predictions with saved network
    for i in range(len(inputs)):
        predicted = nn2.predict(inputs[i])
        print("input: ", inputs[i],
              "output:", outputs[i],
              "predicted: ", predicted)
    #nn.train(inputs, outputs, 10000)
