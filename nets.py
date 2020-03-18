import tensorflow as tf
from tensorflow.keras.layers import Dense
tf.keras.backend.set_floatx('float64')
import numpy as np
# from misc import *

class Q1(tf.keras.Model):
    def __init__(self):
        super(Q1,self).__init__()

        self.l1 = Dense(10, input_shape=(1,),
        kernel_regularizer=tf.keras.regularizers.l1(0.01),
        activity_regularizer=tf.keras.regularizers.l2(0.01),
        kernel_initializer=tf.random_uniform_initializer(),
        bias_initializer = tf.random_uniform_initializer())
        self.l2 = Dense(33,
         kernel_regularizer=tf.keras.regularizers.l1(0.01),
         activity_regularizer=tf.keras.regularizers.l2(0.01),
         kernel_initializer=tf.random_uniform_initializer(),
         bias_initializer = tf.random_uniform_initializer())
        self.l3 = Dense(33, kernel_regularizer=tf.keras.regularizers.l1(0.01),
        activity_regularizer=tf.keras.regularizers.l2(0.01),
        kernel_initializer=tf.random_uniform_initializer(),
        bias_initializer = tf.random_uniform_initializer())
        self.l5 = Dense(1, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01),
    kernel_initializer=tf.random_uniform_initializer(),
    bias_initializer = tf.random_uniform_initializer())

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.dropout(feat,rate=0.1)
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.dropout(feat, rate=0.1)
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.dropout(feat, rate=0.1)
        value = tf.nn.tanh(self.l5(feat))
        return value

    def predict(self, betas):
        inp = np.expand_dims(betas, axis=1)
        return np.squeeze(self(inp).numpy())

    def initialize(self):
        inp = np.expand_dims(np.array([0.]),axis=1)
        self(inp)
        return
    def __str__(self):
        return self.name

class Q2(tf.keras.Model):
    def __init__(self):
        super(Q2,self).__init__()

        self.l1 = Dense(10, input_shape=(2,),kernel_initializer=tf.random_uniform_initializer(),
        bias_initializer = tf.random_uniform_initializer(),
        kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))

        self.l2 = Dense(33, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01),
    kernel_initializer=tf.random_uniform_initializer(),
    bias_initializer = tf.random_uniform_initializer())
        self.l3 = Dense(33, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01),
    kernel_initializer=tf.random_uniform_initializer(),
    bias_initializer = tf.random_uniform_initializer())
        self.l5 = Dense(2, kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01),
    kernel_initializer=tf.random_uniform_initializer(),
    bias_initializer = tf.random_uniform_initializer())

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.dropout(feat,rate=0.1)
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.dropout(feat, rate=0.1)
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.dropout(feat, rate=0.1)
        value = tf.nn.sigmoid(self.l5(feat))
        return value

    def predict(self, history_one):
        inp = np.expand_dims(history_one, axis=0)
        return np.squeeze(self(inp).numpy())

    def initialize(self):
        inp = np.expand_dims(np.array([[0.],[0.]]),axis=1)
        self(inp)
        return
        
    def __str__(self):
        return self.name

class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor,self).__init__()

        self.l1 = Dense(1, input_shape=(1,),
                        kernel_regularizer=tf.keras.regularizers.l1(0.01),
                        activity_regularizer=tf.keras.regularizers.l2(0.01),
                       bias_initializer=tf.random_uniform_initializer())
        self.l2 = Dense(10,
                         kernel_regularizer=tf.keras.regularizers.l1(0.01),
                        activity_regularizer=tf.keras.regularizers.l2(0.01),
                       bias_initializer=tf.random_uniform_initializer())
        self.l3 = Dense(33,
                    kernel_regularizer=tf.keras.regularizers.l1(0.01),
                    activity_regularizer=tf.keras.regularizers.l2(0.01),
                    bias_initializer=tf.random_uniform_initializer())
        self.l5 = Dense(1,
                         kernel_regularizer=tf.keras.regularizers.l1(0.01),
                        activity_regularizer=tf.keras.regularizers.l2(0.01),
                       bias_initializer=tf.random_uniform_initializer())
    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.dropout(feat,rate=0.1)
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.dropout(feat, rate=0.1)
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.dropout(feat, rate=0.1)

        # value = tf.clip_by_value(tf.multiply(10,tf.nn.tanh(self.l5(feat))),clip_value_min=-2, clip_value_max=2)
        value = -tf.nn.sigmoid(self.l5(feat))
        return value

    def prediction(self, betas):
        inp = np.expand_dims(betas, axis=1)
        return np.squeeze(self(inp).numpy())

    def give_action(self):
        dumb = np.array([0.])
        return self(np.expand_dims(dumb,axis=1))


    def initialize(self):
        self.give_action()

    def __str__(self):
        return self.name
