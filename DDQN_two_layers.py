import numpy as np
import basics
import misc
import tensorflow as tf
from tensorflow.keras.layers import Dense
import random

basic = basics.Basics()
basic.define_actions()
actions = basic.actions
ats = misc.make_attenuations(layers=2)

class Memory():
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)
    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)
    @property
    def num_samples(self):
        return len(self._samples)



cardinality_betas = 100

class QN_l1(tf.keras.Model):
    def __init__(self):
        super(QN_l1,self).__init__()
        self.l1 = Dense(30, input_shape=(0,), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l2 = Dense(35, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

        # self.l21 = Dense(90, kernel_initializer='random_uniform',
        #         bias_initializer='random_uniform')
        self.l3 = Dense(cardinality_betas, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        # feat = tf.nn.relu(self.l21(feat))
        value = self.l3(feat)
        return value



class QN_l2(tf.keras.Model):
    def __init__(self):
        super(QN_l2,self).__init__()
        self.l1 = Dense(30, input_shape=(1,2), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l2 = Dense(35, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

        # self.l21 = Dense(90, kernel_initializer='random_uniform',
        #         bias_initializer='random_uniform')
        self.l3 = Dense(cardinality_betas, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        # feat = tf.nn.relu(self.l21(feat))
        value = self.l3(feat)
        return value

class QN_guess(tf.keras.Model):
    def __init__(self):
        super(QN_guess,self).__init__()
        self.l1 = Dense(30, input_shape=(1,4), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l2 = Dense(35, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

        # self.l21 = Dense(90, kernel_initializer='random_uniform',
        #         bias_initializer='random_uniform')
        self.l3 = Dense(2, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        # feat = tf.nn.relu(self.l21(feat))
        value = self.l3(feat)
        return value


#### define the networks #####

qn_l1_prim = QN_l1()
qn_l1_targ = QN_l1()

qn_l2_prim = QN_l2()
qn_l2_targ = QN_l2()

qn_guess_prim = QN_guess()
qn_guess_targ = QN_guess()




def give_first_beta():
    if np.random.random() < epsilon:
        beta1 = np.random.choice(basic.actions[0])
        return beta1
    else:
        input = np.expand_dims(np.array([]), axis=0)
        q1s = qn_l1_prim(input)
        q1s = q1s.numpy()
        label = np.argmax(q1s)
        beta1 = basic.actions[0][label]
        return beta1


def give_second_beta(new_state):
    if np.random.random() < epsilon:
        beta2 = np.random.choice(basic.actions[1])
        return beta2
    else:
        input = np.expand_dims(np.array(new_state), axis=0)
        q2s = qn_l2_prim(input)
        q2s = q2s.numpy()
        label = np.argmax(q2s)
        beta1 = basic.actions[1][label]
        return beta1


def give_guess(new_state):
    if np.random.random() < epsilon:
        guess = np.random.choice(basic.possible_phases,1)[0]
        return guess
    else:
        input = np.expand_dims(np.array(new_state), axis=0)
        qguess = qn_guess_prim(input)
        qguess = qguess.numpy()
        label = np.argmax(guess)
        guess = basic.possible_phases[label]
        return guess


buffer = Memory(10**4)

alpha = .56
states_wasted = 10**4

def main():
    for episode in range(states_wasted):
        phase = np.random.choice([-1,1],1)[0]
        beta1 = give_first_beta()
        p0 = np.exp(-(beta1-(phase*np.cos(ats[0])*alpha))**2)
        outcome1 = np.random.choice([0,1],1,p=[p0,1-p0])[0]
        new_state = [outcome1, beta1]
        beta2 = give_second_beta(new_state)
        p1 = np.exp(-(beta2-(phase*np.sin(ats[0])*alpha))**2)
        outcome2 = np.random.choice([0,1],1,p=[p1,1-p1])[0]
        new_state = [outcome1, outcome2, beta1, beta2]
        guess = give_guess(new_state)
        if guess == phase:
            reward = 1
        else:
            reward = 0
        buffer.add_sample((outcome1, outcome2, beta1, beta2, guess, reward))

main()
buffer.sample(30)
