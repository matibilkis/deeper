import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

class Critic(tf.keras.Model):
    #input_dim: 1 if layer=0, 3 if layer= 2, for the Kennedy receiver ##
    def __init__(self, valreg=0.01, seed_val=0.1, pad_value=-7.):
        super(Critic,self).__init__()

        self.pad_value = pad_value
        self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                  input_shape=(2, 2))
        self.lstm = tf.keras.layers.LSTM(250, return_sequences=True)

        self.l1 = Dense(50,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg), dtype=tf.float32)

        self.l2 = Dense(50, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val), dtype=tf.float32)

        self.l3 = Dense(1, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val), dtype=tf.float32)




    def update_target_parameters(self,primary_net, tau=0.01):
        #### only
        prim_weights = primary_net.get_weights()
        targ_weights = self.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(tau * prim_weights[i] + (1 - tau) * targ_weights[i])
        self.set_weights(weights)
        return

    def call(self, inputs):

        feat = self.mask(inputs)

        feat= self.lstm(feat)
        feat = tf.nn.dropout(feat, rate=0.01)

        feat = tf.nn.relu(self.l1(feat))
        feat = tf.nn.dropout(feat, rate=0.01)

        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.sigmoid(self.l3(feat))
        return feat


    def process_sequence(self,sample_buffer, pad_value = -4., LAYERS=1):
        """" gets data obtained from N experiments: data.shape = (N, 2L+1),
        where +1 accounts for the guess and 2L for (beta, outcome).

        [[a0, o1, a1, o2, a2, o3, a4]
         [same but other experiment]

        ]

        and returns an array of shape (experiments, queries_RNN, 2 ), as accepted by an RNN
        The idea is that i input [\beta, pad_value], and then [outcome, guess].

        Or if I have two layers [\beta, pa_value], [outcome, beta2], [outcome, guess],

        so the number of "queries" to the RNN is layers+1,
        and i'm always interested in putting 2 values more.

        """
        batch_size = sample_buffer.shape[0]
        data = sample_buffer[:,0:(LAYERS+1+1)]
        pad_value = -4.
        padded_data = np.ones((batch_size,LAYERS+1, 2))*pad_value
        padded_data[:,0][:,0] = data[:,0]
        for k in range(1,LAYERS+1):
            padded_data[:,k] = data[:,[k,k+1]]

        rewards_obtained = np.zeros((batch_size, LAYERS+1)).astype(np.float32)
        rewards_obtained[:,-1] = sample_buffer[:,-1]
        return padded_data, rewards_obtained






    @tf.function
    def process_sequence_tf(self, sample_buffer, pad_value = -4., LAYERS=1):
        sample_buffer = tf.convert_to_tensor(experiences.astype(np.float32))
        first = tf.stack([sample_buffer[:,0], pad_value*tf.ones((64,))], axis=-1)
        for k in range(1,LAYERS+1):
            to_stack = tf.stack([sample_buffer[:,k], sample_buffer[:,k+1]], axis=-1)
            first = tf.stack([first, to_stack], axis=1)

        rewards = tf.zeros((sample_buffer.shape[0]))
        rewards = tf.stack([rewards,sample_buffer[:,-1]], axis=-1)
        rewards = tf.expand_dims(rewards, axis=2)
        return first, rewards

    def pad_single_sequence(self, seq, pad_value = -4., LAYERS=1):
        """"
        input: [a0, o1, a1, o2, a2, o3, a4]

        output: [[a0, pad], [o1, a1], [...]]

        the cool thing is that then you can put this to predict the greedy guess/action.
        """
        pad_value = -4.
        padded_data = np.ones((1,LAYERS+1, 2))*pad_value
        padded_data[0][0][0] = seq[0]
        #padded_data[0][0] = data[0]
        for k in range(1,LAYERS+1):
            padded_data[0][k] = seq[k:(k+2)]
        return padded_data


    def give_td_error_Kennedy_guess(self,batched_input,sequential_rews_with_zeros):
        '''
        this function takes a batch with its corresponding labels
        and retrieves what the true labels are according to network
        prodection on next states.

        For instance, my datapoint is [(\beta, pad), (n, guess)]
        and i want [Max_g Q(\beta, n, guess), reward].


        TO DO: extend this to more layers!!!

        So what you want is
        [Max_{a_1} Q(a0, o1, a_1),
        Max_{a_2} Q(a0, o1, a_1, o2, a_2)
        ,...,
        Max_g Q(h, guess)]

        But of course, we can't take the Max_g, so we replace by the target actor's choice !!!
        '''
        b = batched_input.copy()
        ll = sequential_rews_with_zeros.copy()
        preds1 = self(b)
        b[:,1][:,1] = -b[:,1][:,1]
        preds2 = self(b)
        both = tf.concat([preds1,preds2],1)
        maxs = np.squeeze(tf.math.reduce_max(both,axis=1).numpy())
        ll[:,0] = maxs + ll[:,0]
        ll = np.expand_dims(ll,axis=1)
        return ll


    @tf.function
    def give_td_error_Kennedy_guess_tf(self,batched_input,batched_zeroed_reward):
        preds1 = self(batched_input)

        Level1 = tf.unstack(b, axis=1)
        pad, guess = tf.unstack(Level1[1], axis=1)
        new_guess = tf.multiply(guess,-1)
        flipped_guess = tf.stack([Level1[0],tf.stack([pad, new_guess], axis=1)], axis=2)

        preds2 = self(flipped_guess)
        both = tf.concat([preds1,preds2],1)
        maxs = tf.math.reduce_max(both,axis=1)
        batched_zeroed_reward = tf.stack([maxs, batched_zeroed_reward[:,1] ], axis=1)
        return batched_zeroed_reward


    def give_favourite_guess(self,sequence):
        """"sequence should be [[beta, pad], [outcome, guess]] """
        pred_1 = self(sequence)
        sequence[:,1][:,1] = -sequence[:,1][:,1]
        pred_2 = self(sequence)
        both = tf.concat([pred_1,pred_2],1)
        maxs = tf.argmax(both,axis=1)
        guess = (-1)**maxs.numpy()[0][0]
        return guess

experiences= np.load("experiences.npy")
critic = Critic()
b, rews = critic.process_sequence_tf(experiences)
critic.give_td_error_Kennedy_guess_tf(b, rews)
