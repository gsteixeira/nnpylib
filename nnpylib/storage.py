
import json
from nnpylib.networks import NeuralNetwork

def dump_nn(nn:NeuralNetwork):
    """ Takes a NeuralNetwork object and serialize it to json """
    hidden_layers_data = []
    for i, hid in enumerate(nn.hidden_layers):
        data = {
            "bias": nn.hidden_layers[i].bias,
            "deltas": nn.hidden_layers[i].deltas,
            "weights": nn.hidden_layers[i].weights,
            }
        hidden_layers_data.append(data)
    data = {
        "input_layer": {
            "bias": nn.input_layer.bias,
            "deltas": nn.input_layer.deltas,
            "weights": nn.input_layer.weights,
        },
        "hidden_layers": hidden_layers_data,
        "output_layer": {
            "bias": nn.output_layer.bias,
            "deltas": nn.output_layer.deltas,
            "weights": nn.output_layer.weights,
        },
        "learning_rate": nn.learning_rate,
        "method": nn.nonlinear_method,
    }
    return json.dumps(data)

def load_nn(json_data:dict):
    """ Takes a json representing a NN and load it into 
    a NeuralNetwork object
    """
    data = json.loads(json_data)
    input_size = len(data["input_layer"]["bias"])
    output_size = len(data["output_layer"]["bias"])
    hl_sizes = []
    for i, layer in enumerate(data["hidden_layers"]):
        size = len(layer["bias"])
        hl_sizes.append(size)
    lr = data["learning_rate"]
    method = data["method"]
    nn = NeuralNetwork(input_size, output_size, hl_sizes,
                       learning_rate=lr, method=method)
    # input
    nn.input_layer.bias = data["input_layer"]["bias"]
    nn.input_layer.deltas = data["input_layer"]["deltas"]
    nn.input_layer.weights = data["input_layer"]["weights"]
    # output
    nn.output_layer.bias = data["output_layer"]["bias"]
    nn.output_layer.deltas = data["output_layer"]["deltas"]
    nn.output_layer.weights = data["output_layer"]["weights"]
    # hidden
    for i, layer in enumerate(data["hidden_layers"]):
        nn.hidden_layers[i].bias = layer["bias"]
        nn.hidden_layers[i].deltas = layer["deltas"]
        nn.hidden_layers[i].weights = layer["weights"]
    return nn
