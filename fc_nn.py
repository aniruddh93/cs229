
import numpy as np
import tensorflow as tf
from typing import List, Any



class FullyConnectedNN(tf.keras.Model):
    """Provides a fully connected NN model."""

    def __init__(self, num_layers: int, layer_dim: List[int]):
        """Constructs a multi-layer (specified by 'num_layers' linear model with Relu activation.
           Last layer (not included in num_layers)  will have output_dim=1 and sigmoid activation.

        Args:
          num_layers: no. of linear layers in model.
          layer_dim: dim for each layer in model as a list.
        """

        super(FullyConnectedNN, self).__init__()
        self._layers = []
        self._num_layers = num_layers

        for i in range(num_layers):
            l = tf.keras.layers.Dense(units=layer_dim[i], activation='relu', use_bias=True)
            self._layers.append(l)

        last_layer = tf.keras.layers.Dense(units=1, activation='sigmoid', use_bias=True)
        self._layers.append(last_layer)


    def call(self, inputs, training: Any = None, mask: Any = None):
        """Implements forward pass of model."""

        temp = inputs
        
        for l in self._layers:
            out = l(temp)
            temp = out

        return out
            

class LossHistory(tf.keras.callbacks.Callback):
    """Tracks training loss and returns a list containing the loss values during training."""

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
