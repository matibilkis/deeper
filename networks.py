import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np


class QN_l1(tf.keras.Model):
    def __init__(self, betas, dirname_backup_weights= "None"):
        super(QN_l1,self).__init__()
        self.betas = betas
        self.dirname_backup_weights = dirname_backup_weights

        self.l1 = Dense(30, input_shape=(0,), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))
        self.l2 = Dense(35, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))

        self.l3 = Dense(len(self.betas), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')


        # print(self.call(np.expand_dims(np.array([]),axis=0)))
    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        value = self.l3(feat)
        return value

    def give_first_beta(self, epsilon):
        if np.random.random() < epsilon:
            label = np.random.choice(np.arange(len(self.betas)))
            return label, self.betas[label]
        else:
            input = np.expand_dims(np.array([]), axis=0)
            q1s = self.call(input)
            q1s = q1s.numpy()
            label = np.argmax(q1s)
            beta1 = self.betas[label]
            return label, beta1

    def save_now(self):
        tf.keras.models.save_model(self,str(self.dirname_backup_weights))
        return

class QN_l2(tf.keras.Model):
    def __init__(self,betas, dirname_backup_weights= "None"):
        super(QN_l2,self).__init__()
        self.betas = betas
        self.l1 = Dense(30, input_shape=(1,2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))
        self.l2 = Dense(35, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))

        self.l3 = Dense(len(self.betas), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')


    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        # feat = tf.nn.relu(self.l21(feat))
        value = self.l3(feat)
        return value


    def give_second_beta(self,new_state, epsilon):
        if np.random.random() < epsilon:
            label = np.random.choice(np.arange(len(self.betas)))
            return label, self.betas[label]
        else:
            input = np.expand_dims(np.array(new_state), axis=0)
            q2s = self.call(input)
            q2s = q2s.numpy()
            label = np.argmax(q2s)
            beta2 = self.betas[label]
            return label, beta2

    def save_now(self):
        self.save_weigths(str(self.dirname_backup_weights))
        return


class QN_guess(tf.keras.Model):
    def __init__(self,phases, dirname_backup_weights= "None"):
        super(QN_guess,self).__init__()
        self.phases = phases
        self.l1 = Dense(30, input_shape=(1,4), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l2 = Dense(35, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

        self.l3 = Dense(len(self.phases), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        # self.c=0
    def call(self, input):
        # print(self.c)
        # self._set_inputs(input)
        # self.c+=1
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        # feat = tf.nn.relu(self.l21(feat))
        value = self.l3(feat)
        return value


    def give_guess(self,new_state, epsilon):
        if np.random.random() < epsilon:
            guess = np.random.choice(self.phases,1)[0]
            return int((guess+1)/2), guess
        else:
            input = np.expand_dims(np.array(new_state), axis=0)
            qguess = self.call(input)
            guess = qguess.numpy()
            label = np.argmax(guess)
            guess = self.phases[label]
            return int((guess+1)/2), guess
    def save_now(self):
        self.save_weigths(str(self.dirname_backup_weights))
        return
