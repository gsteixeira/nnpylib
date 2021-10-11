# NNPyLib - Yet another Neural Network Python library

NNpylib is a simplistic neural network library for python.

It aims to be simple as possible, to be easy to use and to allow anyone who wishes to understand neural networks can study it.

NNpylib is built in pure python, it doesn't use numpy, to make it clear as possible. It is done for **educational purposes**.

Yet it provides a working reliable neural network that can be used in your applications with minimal effort.

## Features:

- Support for multiple hidden layers.
- Configurable non linear function. Supports: **sigmoid**, **relu**, **leaky relu** and **tanh**
- Built in pure python.
- It is possible to persist the network state, so once trained you can store it and use without training again.

## Instalation

```shell
    pip install nnpylib
```

## usage

Create a neural network.
```python
    # set the parameters
    size_of_input = 2 # how many neurons on the input layer.
    size_of_output = 1 # how many on the output.
    hidden_layers = [4,] # a list with the size of each of hidden layers.
    # create the network
    nn = NewNeuralNetwork(size_of_input, size_of_output, hidden_layers,
                          method="leaky_relu")
    # train it..
    how_many_times = 10000
    nn.train(input_data, expected_output, how_many_times)
    # Now make predictions
    predicted = nn2.predict(foo)
```

Now you can save your network's state and reuse it later.

```python
    from nnpylib.storage import dump_nn, load_nn
    # dump the network state
    data = dump_nn(nn)
    # You can save that to a file or db.
    # Now, create a new network from the saved data.
    nn2 = load_nn(data)
    # make predictions with that saved network
    predicted = nn2.predict(foo)
```

Test it
```shell
    # if you don't have pytest
    pip install pytest
    # test it
    pytest
``` 

For more ideas, look at *samples* dir.
