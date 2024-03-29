
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import tensorflow.keras.backend as bk


class AttentionLayer(Layer):
    """
    注意力层，用于行为特征序列的汇聚：
    input: [None, T, n] T个时间步，n个特征
    output: [None, n] 通过两个Dense层+softmax学出时间步权重score [None, 1, T]，然后经过矩阵乘法运算对各时间步特征加权求和

    """
    def __init__(self,k_dim):
        super(AttentionLayer, self).__init__()
        self.k_dim = k_dim
        self.w = Dense(self.k_dim, activation='relu')
        self.h = Dense(1, activation=None)

    def call(self, inputs, **kwargs): # [None, T, n]
        if bk.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect 3 dimensions" % (bk.ndim(inputs))
            )
        x = self.w(inputs) # [None, T, k_dim]
        x = self.h(x) # [None, T, 1]
        score = tf.nn.softmax(x,axis=1) # [None,T,1]
        score = tf.transpose(score,[0,2,1]) # [None,1,T]
        value = tf.matmul(score, inputs)  # [None, 1, T]x[None, T, n] = [None, 1, n]
        output = tf.reshape(value, [-1, inputs.shape[-1]])  # [None, n]
        return output

