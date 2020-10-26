import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np


class RNN(tf.keras.Model):
    """
    the one from task 2
    """
    def __init__(self,seed_val=0.05,lrr=0.01):
        super(RNN,self).__init__()
        self.lstm = tf.keras.layers.LSTM(30, input_shape=(2,2),return_sequences=True, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))
        self.optimizer = tf.keras.optimizers.Adam(lr=lrr)
        self.l1 = Dense(30, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l2 = Dense(30, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))
        self.l4 = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))


    def call(self, inputs):
        feat = tf.nn.sigmoid(self.lstm(inputs))
        feat = tf.nn.sigmoid(self.l1(feat))
        feat = tf.nn.sigmoid(self.l2(feat))
        feat = tf.nn.sigmoid(self.l4(feat))
        return feat

    @tf.function
    def train_step(self,data, labels, return_gradients=False):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(data)
            loss = tf.keras.losses.MeanSquaredError()(tf.expand_dims(labels,axis=-1),self(data))
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.trainable_variables)
            grads= [tf.clip_by_value(g,-1.,1.) for g in grads]

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        if return_gradients:
            return loss, grads
        return loss
