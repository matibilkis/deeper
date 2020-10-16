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



class RNNC(tf.keras.Model):
    def __init__(self, pad_value=-7., seed_val=0.05,lrr=0.01,return_sequences=True):
        super(RNNC,self).__init__()

        self.lstm = tf.keras.layers.LSTM(30, return_sequences=return_sequences, input_shape=(2,2), kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))
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
    def train_step(self,data, labels, clipping=False, return_gradients=False):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(tf.convert_to_tensor(data))
            loss = tf.losses.mean_squared_error(tf.squeeze(labels),tf.squeeze(preds))
            grads = tape.gradient(loss, self.trainable_variables)
        if clipping:
            grads = [tf.clip_by_value(k,-1,1) for k in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        if return_gradients:
            return loss, grads
        return loss


    #, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
#
