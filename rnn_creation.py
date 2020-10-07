import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic,self).__init__()

        self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                    input_shape=(None, 2))
        self.lstm = tf.keras.layers.LSTM(30, return_sequences=True, input_shape=(None,2))

        self.l1 = Dense(10,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l4 = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))


    def call(self, inputs):
        feat = tf.squeeze(self.mask(tf.expand_dims(inputs, axis=-1)), axis=-1)
        feat= self.lstm(feat)
        feat = tf.nn.sigmoid(self.l1(feat))
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
        if isinstance(sample_buffer, tf.Tensor):
            sample_buffer = sample_buffer.numpy()
        rr = np.ones(sample_buffer.shape)*self.pad_value
        rr[:,1:] = sample_buffer[:,:-1]
        rr = np.reshape(rr, (sample_buffer.shape[0],self.dolinar_layers+1,2))
        #padded_data[:,selff.dolinar_layers] = data[:,[selff.dolinar_layers+1, selff.dolinar_layers+2]]
        rewards_obtained = np.zeros((sample_buffer.shape[0], self.dolinar_layers+1))
        rewards_obtained[:,-1] = sample_buffer[:,-1]
        return rr, rewards_obtained


    @tf.function
    def process_sequence_tf(self, sample_buffer):
        """Ths function just reshapes (and pads) the aray of experiences (transform the vector
        or collection of vectors to a tensor of batchedd inputs and another tensor of zeroed rewards.
        """
        exps = tf.convert_to_tensor(sample_buffer)
        onns = tf.multiply(self.pad_value,tf.ones((sample_buffer.shape[0],1)))
        s1 = tf.concat([onns,exps[:,:-1]], axis=1)
        s1 = tf.reshape(s1, (sample_buffer.shape[0],self.dolinar_layers+1,2))
        rr = tf.concat([tf.zeros((sample_buffer.shape[0], self.dolinar_layers)), tf.expand_dims(exps[:,-1],axis=1)], axis=1)
        return s1, rr



    @tf.function
    def give_td_errors_tf(self,sequences,zeroed_rews):
        """Gives the td_errors, notice we don't use lstm.stateful 'cause we process
        all the sequence (see the ** marks)

        '"""
        if self.nature != "target":
            raise AttributeError("I'm not the target!")

        final_rews = tf.reshape(zeroed_rews[:,-1], (sequences.shape[0],1,1))
        bellman_tds_noguess = self(sequences)[:,1:-1,:] #**

        # phases = tf.range(self.number_phases, dtype=np.float32)

        unstacked = tf.unstack(tf.convert_to_tensor(sequences))
        phases_concs = {}
        for ph in range(self.number_phases):
            phases_concs[str(ph)] = []
        stacked = {}

        for episode in unstacked:
            prefinal = episode[:-1]
            for ph in range(self.number_phases):
                final = tf.expand_dims(tf.stack([tf.unstack(episode[-1])[0], ph], axis=0), axis=0)
                phases_concs[str(ph)].append(tf.concat([prefinal, final], axis=0))
        #
            for ph in range(self.number_phases):
                stacked[str(ph)] = tf.stack(phases_concs[str(ph)], axis=0)

        all_preds = tf.concat([self(stacked[str(ph)]) for ph in range(self.number_phases)], axis=2) #**
        maxs = tf.math.reduce_max(all_preds,axis=2)[:,-1]
        bellman_td = tf.concat([tf.reshape(bellman_tds_noguess,(sequences.shape[0],self.dolinar_layers-1)), tf.reshape(maxs,(sequences.shape[0],1))], axis=1)
        return tf.concat([bellman_td, tf.reshape(zeroed_rews[:,-1], (sequences.shape[0],1))], axis=1)



    def give_favourite_guess(self,hL):
        """
        hL is history (a_0, o1, a_1 ,... o_L)

        outputs: index of the guessed phase, as to be input in env.give_reward, input_network_guess which is this index
        divided by number_phases (clipped input of the network) ///is this relevant/important? ///

        """
        rr = np.random.randn(self.number_phases,2*self.dolinar_layers+1)
        rr[:,:-1] = hL
        rr[:,-1] = np.arange(self.number_phases) #just to keep the value in [0,1], don't know if it's important
        batched_all_guesses = np.reshape(rr[:,[-2,-1]],(self.number_phases, 1, 2))
        predsq = self(batched_all_guesses)
        guess_index = np.squeeze(tf.argmax(predsq, axis=0))
        return guess_index



    @tf.function
    def step_critic_tf(self, batched_input,labels_critic, optimizer_critic):
     with tf.GradientTape() as tape:
         tape.watch(self.trainable_variables)
         preds_critic = self(batched_input)
         loss_critic = tf.keras.losses.MSE(tf.expand_dims(labels_critic, axis=2), preds_critic)
         loss_critic = tf.reduce_mean(loss_critic)
         grads = tape.gradient(loss_critic, self.trainable_variables)
         #tf.print(" dL_dQ", [tf.math.reduce_mean(k).numpy() for k in grads])

         optimizer_critic.apply_gradients(zip(grads, self.trainable_variables))
         return tf.squeeze(loss_critic)


    # @tf.function
    # def critic_grad_tf(self, experiences):
    #     with tf.GradientTape() as tape:
    #         unstacked_exp = tf.unstack(tf.convert_to_tensor(experiences), axis=1)
    #         to_stack = []
    #         actions_wathed_index = []
    #         for index in range(0,experiences.shape[-1]-3,2): # I consider from first outcome to last one (but guess)
    #             actions_wathed_index.append(index)
    #             to_stack.append(tf.reshape(unstacked_exp[index],(experiences.shape[0],1,1)))
    #
    #         actions_indexed = tf.concat(to_stack,axis=1)
    #         tape.watch(actions_indexed)
    #
    #         index_actions=0
    #         watched_exps=[tf.ones((experiences.shape[0],1,1))*self.pad_value]
    #         watched_actions_unstacked = tf.unstack(actions_indexed, axis=1)
    #         for index in range(0,experiences.shape[-1]-1):
    #             if index in actions_wathed_index:
    #                 watched_exps.append(tf.expand_dims(watched_actions_unstacked[index_actions], axis=2))
    #                 index_actions+=1
    #             else:
    #                 watched_exps.append(tf.reshape(unstacked_exp[index],(experiences.shape[0],1,1)))
    #
    #         qvals = self(tf.reshape(tf.concat(watched_exps, axis=2), (experiences.shape[0],self.dolinar_layers+1,2)))
    #
    #         dq_da = tape.gradient(qvals, actions_indexed)
    #         return dq_da


    def critic_grad_tf(self, experiences):
        #check how this works for L>2
        unstacked_exp = tf.unstack(tf.convert_to_tensor(experiences), axis=1)
        to_stack = []
        actions_wathed_index = []
        for index in range(0,experiences.shape[-1]-3,2): # I consider from first outcome to last one (but guess)
            actions_wathed_index.append(index)
            to_stack.append(tf.reshape(unstacked_exp[index],(experiences.shape[0],1,1)))

        actions_indexed = tf.concat(to_stack,axis=1)

        with tf.GradientTape(persistent = True) as tape:
            tape.watch(actions_indexed)

            index_actions=0
            watched_exps=[tf.ones((experiences.shape[0],1,1))*self.pad_value]
            watched_actions_unstacked = tf.unstack(actions_indexed, axis=1)
            for index in range(0,experiences.shape[-1]-1):
                if index in actions_wathed_index:
                    watched_exps.append(tf.expand_dims(watched_actions_unstacked[index_actions], axis=2))
                    index_actions+=1
                else:
                    watched_exps.append(tf.reshape(unstacked_exp[index],(experiences.shape[0],1,1)))

            qvals = self(tf.reshape(tf.concat(watched_exps, axis=2), (experiences.shape[0],self.dolinar_layers+1,2)))
            qvalsunstckd = tf.unstack(qvals, axis=1)[:-1]
            return [tape.gradient(q, actions_indexed) for q in qvalsunstckd]
