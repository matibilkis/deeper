import tensorflow as tf
from tensorflow.keras.layers import Dense
tf.keras.backend.set_floatx('float64')
import numpy as np
# from misc import *

class Q1(tf.keras.Model):
    def __init__(self):
        super(Q1,self).__init__()

        self.l1 = Dense(10, input_shape=(1,), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None), bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None), kernel_regularizer=tf.keras.regularizers.l1(0.05),
    activity_regularizer=tf.keras.regularizers.l1(0.05))
        self.l2 = Dense(33, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None), bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None),kernel_regularizer=tf.keras.regularizers.l1(0.05),
    activity_regularizer=tf.keras.regularizers.l1(0.05))
        self.l3 = Dense(10, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None), bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None), kernel_regularizer=tf.keras.regularizers.l1(0.05),
    activity_regularizer=tf.keras.regularizers.l1(0.05))
        self.l4 = Dense(33, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None), bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None),kernel_regularizer=tf.keras.regularizers.l1(0.05),
    activity_regularizer=tf.keras.regularizers.l1(0.05))
        self.l5 = Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None), bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=.01, seed=None),kernel_regularizer=tf.keras.regularizers.l1(0.05),
    activity_regularizer=tf.keras.regularizers.l1(0.05))

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
#
# def train_net(episode,  pt, network,optimizer, buffer,epochs=10, batch_size=10):
#     pt = pt.copy()
#     if (episode > batch_size):
#         for k in range(epochs): #loop epoch
#             actions_did, rewards = buffer.sample(batch_size)
#             with tf.device("/cpu:0"):
#                 with tf.GradientTape() as tape:
#                     tape.watch(network.trainable_variables)
#                     predictions = network(np.expand_dims(np.array(actions_did),axis=1))
#                     loss_sum = tf.keras.losses.MSE(predictions,np.expand_dims(np.array(rewards),axis=1))
#                     loss = tf.reduce_mean(loss_sum)
#                     grads = tape.gradient(loss, network.trainable_variables)
#                     optimizer.apply_gradients(zip(grads, network.trainable_variables))
#                     pt.append(ps(greedy_action(network,betas,ep=0)[1]))
#
#     else:
#         pt.append(0.5)
#     return pt
