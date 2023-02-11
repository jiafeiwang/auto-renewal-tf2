# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : Embeddings.py

import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model


class Embeds(Model):
    def __init__(self, sparse_feature_info, is_concat=False):
        """
        :param sparse_feature_info: e.y. [{name:'S1', idx:10, 'onehot_dim':12215, 'embed_size':128},
        #                                 {name:'S2', idx:11, 'onehot_dim':327621,'embed_size':128}...]
        :param is_concat: 是否将多个field Embedding之后的输出进行concatenate
        """
        super(Embeds, self).__init__()
        self.sparse_feature_info = sparse_feature_info
        self.is_concat = is_concat
        self.embed_layers = {sfeat['name']: Embedding(sfeat['onehot_dim'], sfeat['embed_size'])
                             for sfeat in self.sparse_feature_info}

    def call(self, inputs, training=None, mask=None):
        embed_x = [self.embed_layers[sfeat['name']](inputs[:, sfeat['idx']]) for sfeat in
                   self.sparse_feature_info]
        embed_x = tf.convert_to_tensor(embed_x)
        output = tf.transpose(embed_x, [1, 0, 2])  # output [None, f, k]
        if self.is_concat:
            output = tf.reshape(output,[-1,output.shape[1]*output.shape[2]])
        return output
