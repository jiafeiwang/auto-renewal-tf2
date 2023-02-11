# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : Deep.py

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

class Deep(Model):
    def __init__(self,hidden_units, activation='relu', dropout=None, output_unit=1):
        super(Deep, self).__init__()
        self.hidden_units = hidden_units
        self.activation=activation
        self.dropout = dropout
        self.output_unit = output_unit

        self.dense_layers = [Dense(h, activation=self.activation) for h in self.hidden_units]
        self.dropout_layer = None
        if self.dropout is not None:
            self.dropout_layer = Dropout(self.dropout)
        self.output_layer = Dense(self.output_unit, activation=None)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        output = self.output_layer(x)
        return output


class ResDeep(Model):
    def __init__(self, hidden_units, activation='relu', dropout=None):
        super(Deep, self).__init__()
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout

    def build(self, input_shape):
        self.dense_layers = [Dense(h, activation=self.activation) for h in self.hidden_units]
        self.dropout_layer = None
        if self.dropout is not None:
            self.dropout_layer = Dropout(self.dropout)
        self.output_layer = Dense(input_shape[-1], activation=None)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        output = self.output_layer(x)

        x = inputs + x
        return tf.nn.relu(output)