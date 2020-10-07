import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

class Critic(tf.keras.Model):
    def __init__(self, pad_value=-7., seed_val=0.05):
        super(Critic,self).__init__()

        #self.lstm = tf.keras.layers.LSTM(30, return_sequences=True, input_shape=(None,2))

        self.l1 = Dense(60)#,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        #bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l2 = Dense(60)#,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        #bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))


        self.l4 = Dense(1)#, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    #bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))


    def call(self, inputs):
        #feat= self.lstm(inputs)
        feat = tf.nn.sigmoid(self.l1(inputs))
        feat = tf.nn.sigmoid(self.l2(feat))
        feat = tf.nn.sigmoid(self.l4(feat))
        return feat
