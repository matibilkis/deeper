import tensorflow as tf
from tensorflow.keras.layers import Dense
tf.keras.backend.set_floatx('float64')
import numpy as np

class Q1(tf.keras.Model):
    def __init__(self):
        super(Q1,self).__init__()

        self.l1 = Dense(10, input_shape=(1,),kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.l2 = Dense(33, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.l3 = Dense(10, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.l4 = Dense(33, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))
        self.l5 = Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.relu(self.l4(feat))
        value = tf.nn.tanh(self.l5(feat))
        return value

    def prediction(self, betas):
        inp = np.expand_dims(betas, axis=1)
        return np.squeeze(self(inp).numpy())
