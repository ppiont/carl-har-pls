import tensorflow as tf
import numpy as np
import pdb
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32')

class neural_net(tf.keras.Model):
    def __init__(self, regularization = 1e-6,num_neurons=16,num_layers=3): #You can choose to have more input here! E.g. number of neurons.
        super(neural_net, self).__init__()

        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.regularization = regularization

        regu = tf.keras.regularizers.l2(self.regularization)

        self.dense = []
        self.batch_norm = []
        for i in range(self.num_layers):
            self.dense.append(tf.keras.layers.Dense(self.num_neurons,activation='relu',use_bias=True,kernel_regularizer = regu))
            self.batch_norm.append(tf.keras.layers.BatchNormalization())

        self.dense_output = tf.keras.layers.Dense(1,activation='linear',use_bias=True,kernel_regularizer = regu)
    #Define the forward propagation
    def call(self, x):

        for i in range(self.num_layers):
            x = self.dense[i](x)
            x = self.batch_norm[i](x)

        x = self.dense_output(x)

        return x
############################################################

