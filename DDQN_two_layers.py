import numpy as np
import basics
import misc
import tensorflow as tf
from tensorflow.keras.layers import Dense
import random

basic = basics.Basics(resolution=.25)
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



cardinality_betas = len(basic.actions[0])

class QN_l1(tf.keras.Model):
    def __init__(self):
        super(QN_l1,self).__init__()
        self.l1 = Dense(30, input_shape=(0,), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))
        self.l2 = Dense(35, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))

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
        self.l1 = Dense(30, input_shape=(1,2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))
        self.l2 = Dense(35, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))

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



def give_first_beta(epsilon):
    if np.random.random() < epsilon:
        label = np.random.choice(np.arange(len(basic.actions[0])))
        return label, basic.actions[0][label]
    else:
        input = np.expand_dims(np.array([]), axis=0)
        q1s = qn_l1_prim(input)
        q1s = q1s.numpy()
        label = np.argmax(q1s)
        beta1 = basic.actions[0][label]
        return label, beta1

def give_second_beta(new_state, epsilon):
    if np.random.random() < epsilon:
        label = np.random.choice(np.arange(len(basic.actions[1])))
        return label, basic.actions[1][label]
    else:
        input = np.expand_dims(np.array(new_state), axis=0)
        q2s = qn_l2_prim(input)
        q2s = q2s.numpy()
        label = np.argmax(q2s)
        beta2 = basic.actions[1][label]
        return label, beta2


def give_guess(new_state, epsilon):
    if np.random.random() < epsilon:
        guess = np.random.choice(basic.possible_phases,1)[0]
        return int((guess+1)/2), guess
    else:
        input = np.expand_dims(np.array(new_state), axis=0)
        qguess = qn_guess_prim(input)
        guess = qguess.numpy()
        label = np.argmax(guess)
        guess = basic.possible_phases[label]
        return int((guess+1)/2), guess




def learn():
    batch_length=32
    batch = buffer.sample(batch_length)

    s_2_batch = np.array([[ v[0], v[2]] for v in batch ] )
    labels_beta1 = np.array([v[4] for v in batch])

    q_2_prim = qn_l2_prim(np.expand_dims(s_2_batch, axis=0))
    q_2_prim = np.squeeze(q_2_prim.numpy())

    opt_a_2_prim = np.argmax(q_2_prim,axis=1)

    update_for_q_1_prim = qn_l1_targ(np.expand_dims(np.array([[] for i in range(len(batch))]), axis=0)) #targ = target
    update_for_q_1_prim = np.squeeze(update_for_q_1_prim, axis=0)
    qlabels_l1 = update_for_q_1_prim.copy()
    qlabels_l1[np.arange(batch_length), labels_beta1] = np.squeeze(qn_l2_targ(np.expand_dims(s_2_batch, axis=0)).numpy())[np.arange(batch_length),opt_a_2_prim]



    with tf.device("/cpu:0"):
        with tf.GradientTape() as tape:
            tape.watch(qn_l1_prim.trainable_variables)
            pred_q_1s = qn_l1_prim(np.expand_dims(np.array([[] for i in range(len(batch))]), axis=0))

            loss_sum =tf.keras.losses.MSE(pred_q_1s, qlabels_l1)
            loss = tf.reduce_mean(loss_sum)

            grads = tape.gradient(loss, qn_l1_prim.trainable_variables)

            optimizer_ql1.apply_gradients(zip(grads, qn_l1_prim.trainable_variables))

    s_2_batch = np.array([[v[0], v[2]] for v in batch])
    s_3_batch = np.array([[v[0], v[1], v[2], v[3]] for v in batch])

    #labels_guess = np.array([v[7] for v in batch])
    labels_action_2 = np.array([v[5] for v in batch])

    q_3_prim = qn_guess_prim(np.expand_dims(s_3_batch, axis=0))
    q_3_prim = np.squeeze(q_3_prim.numpy())

    opt_a_3_prim = np.argmax(q_3_prim, axis=1)

    update_for_q_2_prim = qn_l2_targ(np.expand_dims(s_2_batch, axis=0))
    update_for_q_2_prim = np.squeeze(update_for_q_2_prim, axis=0)
    qlabels_l2 = update_for_q_2_prim.copy()
    qlabels_l2[np.arange(batch_length), labels_action_2] = np.squeeze(qn_guess_targ(np.expand_dims(s_3_batch, axis=0)).numpy())[np.arange(batch_length), opt_a_3_prim]


    with tf.device("/cpu:0"):
        with tf.GradientTape() as tape:
            tape.watch(qn_l2_prim.trainable_variables)
            pred_q_2s = qn_l2_prim(np.expand_dims(s_2_batch, axis=0))
            loss_sum =tf.keras.losses.MSE(pred_q_2s, qlabels_l2)
            loss = tf.reduce_mean(loss_sum)

            grads = tape.gradient(loss, qn_l2_prim.trainable_variables)
            optimizer_ql2.apply_gradients(zip(grads, qn_l2_prim.trainable_variables))


    s_3_batch = np.array([[v[0], v[1], v[2], v[3]] for v in batch])
    rewards = np.array([v[-1] for v in batch])
    labels_guess = np.array([v[7] for v in batch])

    update_for_q_3_prim = qn_guess_targ(np.expand_dims(s_3_batch, axis=0))
    update_for_q_3_prim = np.squeeze(update_for_q_3_prim, axis=0)
    qlabels_l3 = update_for_q_3_prim.copy()
    qlabels_l3[np.arange(batch_length), labels_guess] = rewards[np.arange(batch_length)]


    with tf.device("/cpu:0"):
        with tf.GradientTape() as tape:
            tape.watch(qn_guess_prim.trainable_variables)
            pred_q_3s = qn_guess_prim(np.expand_dims(s_3_batch, axis=0))
            loss_sum =tf.keras.losses.MSE(pred_q_3s, qlabels_l3)
            loss = tf.reduce_mean(loss_sum)

            grads = tape.gradient(loss, qn_guess_prim.trainable_variables)
            optimizer_ql3.apply_gradients(zip(grads, qn_guess_prim.trainable_variables))

    for t, e in zip(qn_l1_targ.trainable_variables, qn_l1_prim.trainable_variables):
        t.assign(t*(1-TAU) + e*TAU)

    for t, e in zip(qn_l2_targ.trainable_variables, qn_l2_prim.trainable_variables):
        t.assign(t*(1-TAU) + e*TAU)

    for t, e in zip(qn_guess_targ.trainable_variables, qn_guess_prim.trainable_variables):
        t.assign(t*(1-TAU) + e*TAU)
    return



buffer = Memory(10**4)
optimizer_ql1 = tf.keras.optimizers.Adam(lr=0.001)
optimizer_ql2 = tf.keras.optimizers.Adam(lr=0.001)
optimizer_ql3 = tf.keras.optimizers.Adam(lr=0.001)

alpha = .56
states_wasted = 10**3
TAU = 0.08

cum_rews = []
def main():
    for episode in range(states_wasted):
        epsilon = np.exp(-0.001*episode)
        phase = np.random.choice([-1,1],1)[0]
        labelbeta1, beta1 = give_first_beta(epsilon)
        p0 = np.exp(-(beta1-(phase*np.cos(ats[0])*alpha))**2)
        outcome1 = np.random.choice([0,1],1,p=[p0,1-p0])[0]
        new_state = [outcome1, beta1]
        labelbeta2, beta2 = give_second_beta(new_state,epsilon)
        p1 = np.exp(-(beta2-(phase*np.sin(ats[0])*alpha))**2)
        outcome2 = np.random.choice([0,1],1,p=[p1,1-p1])[0]
        new_state = [outcome1, outcome2, beta1, beta2]
        label_guess, guess = give_guess(new_state,epsilon)
        if guess == phase:
            reward = 1
        else:
            reward = 0
        buffer.add_sample((outcome1, outcome2, beta1, beta2, labelbeta1, labelbeta2, guess, label_guess, reward))
        if episode > 10**2:
            learn()
        cum_rews.append(np.sum(cum_rews)+reward)
    return cum_rews/np.arange(1, states_wasted+1)
main()
