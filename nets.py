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

#
# class RNNC(tf.keras.Model):
#     """
#     the one from task 2
#     """
#     def __init__(self,seed_val=0.05,lrr=0.01,lmode=1):
#         super(RNNC,self).__init__()
#         self.lmode=lmode
#         self.lstm = tf.keras.layers.LSTM(30, input_shape=(2,2),return_sequences=True, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))
#         self.optimizer = tf.keras.optimizers.Adam(lr=lrr)
#         self.l1 = Dense(30, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))
#
#         self.l2 = Dense(30, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))
#         self.l4 = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))
#
#
#     def call(self, inputs):
#         feat = tf.nn.sigmoid(self.lstm(inputs))
#         feat = tf.nn.sigmoid(self.l1(feat))
#         feat = tf.nn.sigmoid(self.l2(feat))
#         feat = tf.nn.sigmoid(self.l4(feat))
#         return feat
#
#     @tf.function
#     def train_step(self,data, labels, return_gradients=False):
#         with tf.GradientTape() as tape:
#             tape.watch(self.trainable_variables)
#             preds = self(data)
#             if self.lmode == 1:
#                 loss = tf.losses.mean_squared_error(labels,tf.squeeze(self(data)))
#             elif self.lmode==2:
#                 loss = tf.losses.mean_squared_error(tf.expand_dims(labels,axis=2),self(data))
#             else:
#                 raise AttributeError("what the loss: got {}, type {}".format(self.lmode, type(self.lmode)))
#                 return
#             grads = tape.gradient(loss, self.trainable_variables)
#             grads= [tf.clip_by_value(g,-1.,1.) for g in grads]
#         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
#         if return_gradients:
#             return loss, grads
#         return loss
#
#

class RNNnight(tf.keras.Model):
    """
    the one from task 2
    """
    def __init__(self,seed_val=0.05,lrr=0.01):
        super(RNNnight,self).__init__()
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
            loss = tf.keras.losses.MeanSquaredError(tf.expand_dims(labels,axis=-1),self(data))
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.trainable_variables)
            grads= [tf.clip_by_value(g,-1.,1.) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        if return_gradients:
            return loss, grads
        return loss





class RNNC(tf.keras.Model):
    """
    the one from task 2
    """
    def __init__(self,seed_val=0.05,lrr=0.01,lmode=1):
        super(RNNC,self).__init__()
        self.lmode=lmode
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
            if self.lmode == 1:
                loss = tf.losses.mean_squared_error(labels,tf.squeeze(self(data)))
            elif self.lmode==2:
                loss = tf.losses.mean_squared_error(tf.expand_dims(labels,axis=2),self(data))
            elif self.lmode==3:
                loss = tf.keras.losses.MSE(tf.expand_dims(labels,axis=-1),self(data))
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, self.trainable_variables)
            grads= [tf.clip_by_value(g,-1.,1.) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        if return_gradients:
            return loss, grads
        return loss



class task1(tf.keras.Model):
    def __init__(self, pad_value=-7., seed_val=0.05,lrr=0.01,return_sequences=True):
        super(task1,self).__init__()
        """
        the one of task1
        """
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
