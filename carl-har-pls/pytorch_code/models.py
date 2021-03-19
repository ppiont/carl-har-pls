import numpy as np
import pdb
import torch.nn as nn


# Define a neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, n_input=50, num_neurons=16, num_layers=3,
                 activation=nn.ReLU()):
        """Initialization"""
        super(NeuralNetwork, self).__init__()

        # Network parameters
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.n_input = n_input
        self.activation = activation

        # First layer
        self.in_layer = nn.Linear(self.n_input, self.num_neurons)

        # Remaining dense and batchnorm layers
        self.dense = []
        self.batchnorm = []
        for i in range(self.num_layers - 1):
            self.dense.append(nn.Linear(self.num_neurons, self.num_neurons))
            self.batchnorm.append(nn.BatchNorm1d(self.num_neurons))

        # Output layer
        self.out_layer = nn.Linear(self.num_neurons, 1)

    def forward(self, x):
        """Forward propagation"""

        x = self.activation(self.in_layer(x))

        for i in range(self.num_layers - 1):
            x = self.activation(self.dense[i](x))
            x = self.batchnorm[i](x)

        return self.out_layer(x)