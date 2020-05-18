import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

class Critic(tf.keras.Model):
    def __init__(self,nature, valreg=0.01, seed_val=0.3, pad_value=-7., dolinar_layers=2, tau=0.01, number_phases=2):
        '''
        dolinar_layers= number of photodetections
        pad_value: value not considered by the lstm
        valreg: regularisation value
        seed_val: interval of random parameter inizialitaion.
        nature: primary or target
        '''
        super(Critic,self).__init__()

        self.pad_value = pad_value
        self.number_phases = number_phases
        self.nature = nature
        self.dolinar_layers = dolinar_layers
        self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                  input_shape=(self.dolinar_layers, 2)) #(beta1, pad), (n1, beta2), (n2, guess). In general i will have (layer+1)
        self.lstm = tf.keras.layers.LSTM(500, return_sequences=True)

        self.tau = tau
        self.l1 = Dense(250,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg))

        self.l2 = Dense(100, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l3 = Dense(100, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l4 = Dense(1, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))



    def update_target_parameters(self,primary_net):
        #### only
        # for i,j in zip(self.get_weights(), primary_net.get_weights()):
        #     tf.assign(i, tau*j + (i-tau)*i )
        prim_weights = primary_net.get_weights()
        targ_weights = self.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(self.tau * prim_weights[i] + (1 - self.tau) * targ_weights[i])
        self.set_weights(weights)
        return

    def call(self, inputs):
        feat = self.mask(inputs)
        feat= self.lstm(feat)
        # feat = tf.nn.dropout(feat, rate=0.01)
        feat = tf.nn.relu(self.l1(feat))
        # feat = tf.nn.dropout(feat, rate=0.01)
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.sigmoid(self.l4(feat))
        return feat


    def process_sequence(self,sample_buffer):
        """"
        sample_buffer: array of shape (N,2*self.layers +1), N>1 (+1 for the reward)

        gets data obtained from N experiments: data.shape = (N, 2L+1),
        where +1 accounts for the guess and 2L for (beta, outcome).

        [[a0, o1, a1, o2, a2, o3, a4]
         [same but other experiment]
        ]

        and returns an array of shape (experiments, self.layers, 2 ), as accepted by an RNN
        """
        rr = np.ones(sample_buffer.shape)*self.pad_value
        rr[:,1:] = sample_buffer[:,:-1]
        rr = np.reshape(rr, (sample_buffer.shape[0],self.dolinar_layers+1,2))
        #padded_data[:,selff.dolinar_layers] = data[:,[selff.dolinar_layers+1, selff.dolinar_layers+2]]
        rewards_obtained = np.zeros((sample_buffer.shape[0], self.dolinar_layers+1))
        rewards_obtained[:,-1] = sample_buffer[:,-1]
        return rr, rewards_obtained


    # def pad_single_sequence(self, seq):
    #     """"
    #     input: [a0, o1, a1, o2, a2, o3, a4]
    #
    #     output: [[a0, pad], [o1, a1], [...]]
    #
    #     the cool thing is that then you can put this to predict the greedy guess/action.
    #     """
    #     padded_data = np.ones((1,self.dolinar_layers+1, 2))*self.pad_value
    #     padded_data[0][0][0] = seq[0]
    #     #padded_data[0][0] = data[0]
    #     for k in range(1,self.dolinar_layers+1):
    #         padded_data[0][k] = seq[k:(k+2)]
    #     return padded_data

    def give_td_error_Kennedy_guess(self,batched_input,sequential_rews_with_zeros):
        # this function takes as input the actions as given by the target actor (but the first one!)
        #and outpus the correspoindg TD-errors for DDPG! To obtain them from sample of buffer
        #you call the method targeted_sequence from the actor_target and then the process_sequence
        #of this critic network.
        if self.nature != "target":
            raise AttributeError("I'm not the target!")
            return
        b = batched_input.copy()
        ll = sequential_rews_with_zeros.copy()
        for k in range(self.dolinar_layers):
            ll[:,k] = np.squeeze(self(b))[:,k+1] + ll[:,k]

        b[:,-1][:,-1] = 0.
        all_preds = self(b)
        for phase in np.arange(1,self.number_phases)/self.number_phases:
            b[:,-1][:,-1] = phase
            all_preds = tf.concat([all_preds,self(b)],2)

        maxs = np.squeeze(tf.math.reduce_max(all_preds,axis=2).numpy())
        ll[:,-2] = maxs[:,-1] # This is the last befre the guess. So the label is max_g Q(h-L, g)
        ll = np.expand_dims(ll,axis=1)
        return ll

    def give_favourite_guess(self,hL):
        """
        hL is history (a_0, o1, a_1 ,... o_L)

        outputs: index of the guessed phase, as to be input in env.give_reward, input_network_guess which is this index
        divided by number_phases (clipped input of the network) ///is this relevant/important? ///

        """
        rr = np.random.randn(self.number_phases,2*self.dolinar_layers+1)
        rr[:,:-1] = hL
        rr[:,-1] = np.arange(self.number_phases)/self.number_phases #just to keep the value in [0,1], don't know if it's important
        batched_all_guesses = np.reshape(rr[:,[-2,-1]],(self.number_phases, 1, 2))
        predsq = self(batched_all_guesses)
        guess = np.squeeze(tf.argmax(predsq, axis=0))
        input_netork_guess = guess/self.number_phases
        return guess, input_netork_guess





##### ACTOR CLASSS ####
class Actor(tf.keras.Model):
    def __init__(self, nature, valreg=0.01, seed_val=0.1, pad_value = -7.,
                 dolinar_layers=2,tau=0.01):
        super(Actor,self).__init__()
        self.dolinar_layers = dolinar_layers
        self.pad_value = pad_value
        self.nature = nature
        self.tau = tau

        if nature == "primary":
            self.lstm = tf.keras.layers.LSTM(500, return_sequences=True, stateful=True)
            self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                  input_shape=(1,1))#CHECK
        elif nature == "target":
            self.lstm = tf.keras.layers.LSTM(500, return_sequences=True, stateful=False)
            self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                  input_shape=(self.dolinar_layers, 1)) #'cause i feed altoghether.
        else:
            print("Hey! the character is either primary or target")
        self.l1 = Dense(250,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg), dtype='float32')

        self.l2 = Dense(100, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val), dtype='float32')

        self.l3 = Dense(100, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val), dtype='float32')

        self.l4 = Dense(1, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val), dtype='float32')



    def update_target_parameters(self,primary_net):
        #### only
        # for i,j in zip(self.get_weights(), primary_net.get_weights()):
        #     tf.assign(i, tau*j + (i-tau)*i )
        prim_weights = primary_net.get_weights()
        targ_weights = self.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(self.tau * prim_weights[i] + (1 - self.tau) * targ_weights[i])
        self.set_weights(weights)
        return

    def call(self, inputs):
        feat = self.mask(inputs)
        feat= self.lstm(feat)
        # feat = tf.nn.dropout(feat, rate=0.01)
        feat = tf.nn.relu(self.l1(feat))
        # feat = tf.nn.dropout(feat, rate=0.01)
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.tanh(self.l4(feat))

        return feat

    def process_sequence_of_experiences(self, experiences):

        export = experiences.copy()
        for index in range(1,2*self.dolinar_layers-1,2): # I consider from first outcome to last one (but guess)
            export[:,index+1] = np.squeeze(self(np.reshape(np.array(export[:,index]),
                                                                 (experiences.shape[0],1,1))))
        return export

    def __str__(self):
        return self.name
