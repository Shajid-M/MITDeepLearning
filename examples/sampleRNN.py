import tensorflow as tf

class MyRNN(tf.keras.layers.Layer):
    def __init__(self, rnn_units, inp_dim, out_dim):
        super(MyRNN, self).__intit__()

        self.w_xh = self.add_weight([rnn_units, inp_dim])
        self.w_hh = self.add_weight([rnn_units, rnn_units])
        self.w_hy = self.add_weight([out_dim, rnn_units])
                                    
        self.h = tf.zeros([rnn_units, 1])

    def call(self, x):
        self.h = tf.math.tanh(self.w_hh * self.h + self.w_xh * x)
        output = self.w_hy * self.h
        return output, self.h
    