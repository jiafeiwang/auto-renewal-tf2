# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : PNN.py

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from layers.products import InnerProductLayer
from base.Embeddings import Embeds
from base.Deep import Deep, ResDeep
from layers.attentions import AttentionLayer


class JointNNClassifier(Model):
    def __init__(self, dense_feature_info=None, sparse_feature_info=None, behavior_feature_info=None,
                 hidden_units=None, ak_dim=None, output_unit=31, dropout=0.2):
        super(JointNNClassifier, self).__init__()
        self.dense_feature_info = dense_feature_info
        self.sparse_feature_info = sparse_feature_info
        self.behavior_feature_info = behavior_feature_info
        self.hidden_units = hidden_units
        self.ak_dim = ak_dim
        self.output_unit = output_unit
        self.dropout = dropout

    def build(self, input_shape):
        self.normal_embeds = Embeds(self.sparse_feature_info)
        self.behavior_embeds = [Embeds(t,True) for t in self.behavior_feature_info]
        self.deep = Deep(self.hidden_units, self.dropout)
        self.inner_product_layer = InnerProductLayer()
        self.attention_layer = AttentionLayer(self.ak_dim)
        self.output_layer = Dense(self.output_unit)

    def call(self, inputs, training=None, mask=None):
        embed_normal = self.embeds(inputs)
        embed_normal = tf.reshape(embed_normal, [-1, embed_normal.shape[1] * embed_normal.shape[2]])
        product = self.inner_product_layer(embed_normal)

        dense_x = tf.gather(inputs, [dfeat['idx'] for dfeat in self.dense_feature_info], axis=1)

        behavior_x = []
        for t_embed in self.behavior_embeds:
            behavior_x.append(behavior_x)
        behavior_x = tf.transpose(tf.convert_to_tensor(behavior_x),[1,0,2])
        behavior_x = self.attention_layer(behavior_x)

        x = tf.concat([dense_x, embed_normal, product, behavior_x])
        x = self.deep(x)

        output = self.output_layer(x)
        return tf.nn.softmax(output,axis=1)
