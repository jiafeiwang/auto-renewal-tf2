
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import tensorflow.keras.backend as bk


class AttentionLayer(Layer):
    def __init__(self,k_dim):
        super(AttentionLayer, self).__init__()
        self.k_dim = k_dim

    def build(self, input_shape):
        self.w = Dense(self.k_dim,activation='relu')
        self.h = Dense(1,activation=None)

    def call(self, inputs, **kwargs): #[None, T, k]
        if bk.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect 3 dimensions" % (bk.ndim(inputs))
            )
        x = self.w(inputs) #[None, T, k_dim]
        x = self.h(x) #[None, T, 1]
        score = tf.nn.softmax(x,axis=1) #[None,T,1]
        score = tf.transpose(score,[0,2,1]) #[None,1,T]
        value = tf.matmul(score, inputs)  # [None, 1, T]x[None, T, k] = [None, 1, k]
        output = tf.reshape(value, [-1, inputs.shape[-1]])  # [None, k]
        return output

