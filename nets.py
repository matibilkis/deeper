import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

class Critic(tf.keras.Model):
    def __init__(self,nature, seed_val=.1, pad_value=-7., dolinar_layers=2, tau=0.01, number_phases=2, dropout_rate=0.0):
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
        self.dropout_rate = dropout_rate
        self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                    input_shape=(2, 2))
                                  # input_shape=(self.dolinar_layers, 2)) #(beta1, pad), (n1, beta2), (n2, guess). In general i will have (layer+1)
        self.lstm = tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(None,2)) #input_shape = (time_steps, features)

        self.tau = tau
        self.l1 = Dense(250,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l2 = Dense(300, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l3 = Dense(300, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l4 = Dense(1, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
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
        feat = tf.squeeze(self.mask(tf.expand_dims(inputs, axis=-1)), axis=-1)
        feat= self.lstm(feat)
        #feat = tf.nn.dropout(feat, rate=self.dropout_rate)
        feat = self.l1(feat)
        #feat = tf.nn.dropout(feat, rate=self.dropout_rate)
        # feat = tf.nn.tanh(self.l2(feat))
        feat = tf.nn.sigmoid(self.l3(feat))
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
            qvalsunstckd = tf.unstack(qvals, axis=1)
            return [tape.gradient(q, actions_indexed) for q in qvalsunstckd]


##### ACTOR CLASSS ####
class Actor(tf.keras.Model):
    def __init__(self, nature, seed_val=0.01, pad_value = -7.,
                 dolinar_layers=2,tau=0.01,dropout_rate=0.0):
        super(Actor,self).__init__()
        self.dolinar_layers = dolinar_layers
        self.pad_value = pad_value
        self.nature = nature
        self.tau = tau

        self.output_lstm = 250
        self.lstm = tf.keras.layers.LSTM(self.output_lstm, return_sequences=True, stateful=True, input_shape=(1,1))
        self.mask = tf.keras.layers.Masking(mask_value=pad_value, input_shape=(1,1))


        self.l1 = Dense(30,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val), dtype='float32')

        self.l2 = Dense(10, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val), dtype='float32')

        self.l3 = Dense(10, kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val), dtype='float32')

        self.l4 = Dense(1,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val), dtype='float32')


    def reset_states_workaround(self, new_batch_size=1):
        self.lstm.states = [tf.Variable(tf.zeros((new_batch_size,self.output_lstm))), tf.Variable(tf.zeros((new_batch_size,self.output_lstm)))]
        self.lstm.input_spec = [tf.keras.layers.InputSpec(shape=(new_batch_size,None,1), ndim=3)]
        return


    def update_target_parameters(self,primary_net):
        prim_weights = primary_net.get_weights()
        targ_weights = self.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(self.tau * prim_weights[i] + (1 - self.tau) * targ_weights[i])
        self.set_weights(weights)
        return

    def call(self, inputs):
        feat = tf.squeeze(self.mask(tf.expand_dims(inputs, axis=-1)), axis=-1)
        feat= self.lstm(inputs)
        #feat = tf.nn.dropout(feat, rate=self.dropout_rate)
        feat = tf.nn.sigmoid(self.l1(feat))
        # feat = tf.nn.dropout(feat, rate=self.dropout_rate)
        feat = tf.nn.sigmoid(self.l2(feat))
        # feat = tf.nn.sigmoid(self.l3(feat))
        feat = tf.nn.sigmoid(self.l4(feat))
        # feat = tf.clip_by_value(feat, 0.0, 1.0)
        return feat

    # def process_sequence_of_experiences(self, experiences):
    #     export = experiences.copy()
    #     for index in range(1,2*self.dolinar_layers-1,2): # I consider from first outcome to last one (but guess)
    #         export[:,index+1] = np.squeeze(self(np.reshape(np.array(export[:,index]),
    #                                                              (experiences.shape[0],1,1))))
    #
    #     return export

    @tf.function
    def process_sequence_of_experiences_tf(self, experiences, who):
        #### this is trivial for dolianr_layers = 1 #####
        unstacked_exp = tf.unstack(tf.convert_to_tensor(experiences), axis=1)
        to_stack = []
        for index in range(2*self.dolinar_layers-1): # I consider from first outcome to last one (but guess)
            if (index==0):
                if who == "target":
                    to_stack.append(unstacked_exp[index])
                else:
                    to_stack.append(tf.squeeze(self(tf.ones((experiences.shape[0], 1, 1))*self.pad_value)))
            if (index%2 == 1):
                to_stack.append(unstacked_exp[index])
                to_stack.append(tf.squeeze(self(tf.reshape(unstacked_exp[index],(experiences.shape[0],1,1)))))
        for index in range(2*self.dolinar_layers-1, 2*self.dolinar_layers+2):
            to_stack.append(unstacked_exp[index])

        if self.dolinar_layers>1:
            self.reset_states() #otherwise is like not doing anything so no problem :-)
        else:
            tf.reshape(unstacked_exp[1],(experiences.shape[0],1,1)) #just call it once to initialize
        return tf.stack(to_stack, axis=1)

    @tf.function
    def actor_grad_tf(self, dq_da, experiences, optimizer_actor):
        finns = [tf.ones((experiences.shape[0], 1,1))*self.pad_value]
        unstacked_exp = tf.unstack(experiences, axis=1)
        for index in range(1,2*self.dolinar_layers-2,2):
            finns.append(tf.reshape(unstacked_exp[index], (experiences.shape[0], 1,1)))
        final_preds = tf.concat(finns, axis=1)
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            final_preds = self(final_preds)
        dq_da = tf.multiply(dq_da, -1/samples.shape[0])
        da_dtheta=tape.gradient(final_preds, self.trainable_variables, output_gradients=dq_da) #- because you wanna minimize, and the Q value maximizes..
            #/experiences.shape[0] because it's 1/N (this is checked in debugging actor notebook... proof of [...])
        optimizer_actor.apply_gradients(zip(da_dtheta, self.trainable_variables))
        return

    def __str__(self):
        return self.name
