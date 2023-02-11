# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : joint.py

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from layers.products import InnerProductLayer
from base.Embeddings import Embeds
from base.Deep import Deep, ResDeep
from layers.attentions import AttentionLayer


class DNNClassifier(Model):
    def __init__(self, dense_feature_info=None, sparse_feature_info=None, behavior_feature_info=None,
                 hidden_unit=None, att_dim=None, output_unit=31, dropout=0.2):
        """
        :param dense_feature_info: e.y. [{name:'D1', idx:0}, {name:'D2', idx:1}...]
        :param sparse_feature_info:  e.y. [{name:'S1', idx:10, 'onehot_dim':12215, 'embed_size':128},
        #                            {name:'S2', idx:11, 'onehot_dim':327621,'embed_size':128}...]
        :param behavior_feature_info:  e.y. [[{name:'BS1', idx:20, 'onehot_dim':1000, 'embed_size':128},
        #                               {name:'S2', idx:21, 'onehot_dim':2000,'embed_size':128}...],
                                       [{name:'BS1', idx:20, 'onehot_dim':1000, 'embed_size':128},
        #                               {name:'S2', idx:21, 'onehot_dim':2000,'embed_size':128}...]...]

        :param hidden_unit: hidden units of Deep(MLP)
        :param att_dim: hidden units of attention layer
        :param output_unit: output unit
        :param dropout: dropout rate of MLP
        """
        super(DNNClassifier, self).__init__()
        self.dense_feature_info = dense_feature_info
        self.sparse_feature_info = sparse_feature_info
        self.behavior_feature_info = behavior_feature_info

        self.hidden_unit = hidden_unit
        self.att_dim = att_dim
        self.output_unit = output_unit
        self.dropout = dropout

        self.common_embeds = Embeds(self.sparse_feature_info)  # embeds for common sparse feat
        self.seq_embeds = [Embeds(t, True) for t in self.behavior_feature_info]  # embeds for sequential features
        self.deep = Deep(self.hidden_unit, self.dropout)  # MLP
        self.inner_product_layer = InnerProductLayer()  # inner product layer for common sparse features
        self.attention_layer = AttentionLayer(self.att_dim)  # attention layer for sequential features
        self.output_layer = Dense(self.output_unit)  # output layer

    def call(self, inputs, training=None, mask=None):
        common_embed_x = self.embeds(inputs)  # [None, n, k]
        product = self.inner_product_layer(common_embed_x)  # [None, n*(n-1)/2]
        common_embed_x = tf.reshape(common_embed_x,
                                    [-1, common_embed_x.shape[1] * common_embed_x.shape[2]])  # [None, n*k]

        dense_x = tf.gather(inputs, [dfeat['idx'] for dfeat in self.dense_feature_info], axis=1)

        behavior_x = []
        for t_embed in self.seq_embeds:
            behavior_x.append(behavior_x)
        behavior_x = tf.transpose(tf.convert_to_tensor(behavior_x),[1,0,2])  # [None, T, m]
        behavior_x = self.attention_layer(behavior_x) # [None, m]

        x = tf.concat([dense_x, common_embed_x, product, behavior_x])  # [None, m+n*k+...]
        x = self.deep(x)

        output = self.output_layer(x) # [None, 31]
        return tf.nn.softmax(output,axis=1) # [None, 31]
