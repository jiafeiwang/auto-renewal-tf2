# -*- coding:utf-8 -*-
# @Author   : Wang Jiahui
# @File     : product.py
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as bk


class InnerProductLayer(Layer):
    """
    计算特征内积
    """
    def __init__(self):
        super(InnerProductLayer, self).__init__()

    def call(self, inputs, **kwargs):
        if bk.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect 3 dimensions" % (bk.ndim(inputs))
            )

        field_num = inputs.shape[1]

        idx_i, idx_j = [], []
        for i in range(field_num-1):
            for j in range(i + 1, field_num):
                idx_i.append(i)
                idx_j.append(j)
        i_interaction = tf.gather(inputs, idx_i, axis=1)
        j_interaction = tf.gather(inputs, idx_j, axis=1)
        product = tf.reduce_sum(i_interaction * j_interaction, axis=-1, keepdims=False)
        return product


class OuterProductLayer(Layer):
    """
    计算特征外积
    """
    def __init__(self, reg_w):
        super(OuterProductLayer, self).__init__()
        self.reg_w = reg_w

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected input_shape %d, expect 3 dimensions" % (len(input_shape))
            )
        self.field_num = input_shape[1]
        self.k = input_shape[2]
        self.interact_num = self.field_num * (self.field_num - 1) // 2

        self.w = self.add_weight(name='w', shape=(self.k, self.interact_num, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=tf.keras.regularizers.l2(self.reg_w),
                                 trainable=True)

    def call(self, inputs, **kwargs):  # [None, field, k]
        if bk.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect 3 dimensions" % (bk.ndim(inputs))
            )

        idx_i, idx_j = [], []
        for i in range(self.field_num):
            for j in range(i + 1, self.field_num):
                idx_i.append(i)
                idx_j.append(j)

        i_interaction = tf.gather(inputs, idx_i, axis=1)  # [None, interact_num, k]
        j_interaction = tf.gather(inputs, idx_j, axis=1)  # [None, interact_num, k]
        i_interaction = tf.expand_dims(i_interaction, axis=1)  # [None, 1, interact_num, k] 忽略掉第一维，需要两维与w一致才能进行点乘

        tmp = tf.multiply(i_interaction,
                          self.w)  # [None, 1, interact_num, k]x[k, interact_num, k] = [None, k, interact_num, k]
        tmp = tf.reduce_sum(tmp, axis=-1)  # [None, k, interact_num]
        tmp = tf.multiply(tf.transpose(tmp, [0, 2, 1]), j_interaction)  # [None, interact_num, k]
        product = tf.reduce_sum(tmp, axis=-1)  # [None, interact_num]
        return product
