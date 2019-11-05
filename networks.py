import tensorflow as tf
from tensorflow.keras.layers import Dense

class QN_l1(tf.keras.Model):
    def __init__(self, cardinality_betas):
        super(QN_l1,self).__init__()
        self.l1 = Dense(30, input_shape=(0,), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))
        self.l2 = Dense(35, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))

        self.l3 = Dense(cardinality_betas, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        value = self.l3(feat)
        return value

class QN_l2(tf.keras.Model):
    def __init__(self,cardinality_betas):
        super(QN_l2,self).__init__()
        self.l1 = Dense(30, input_shape=(1,2), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))
        self.l2 = Dense(35, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None),
                bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))


        self.l3 = Dense(cardinality_betas, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        # feat = tf.nn.relu(self.l21(feat))
        value = self.l3(feat)
        return value

class QN_guess(tf.keras.Model):
    def __init__(self,number_phases=2):
        super(QN_guess,self).__init__()
        self.l1 = Dense(30, input_shape=(1,4), kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        self.l2 = Dense(35, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

        self.l3 = Dense(number_phases, kernel_initializer='random_uniform',
                bias_initializer='random_uniform')

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l2(feat))
        # feat = tf.nn.relu(self.l21(feat))
        value = self.l3(feat)
        return value
