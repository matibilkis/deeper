import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

class Critic(tf.keras.Model):
    def __init__(self,nature, valreg=0.01, seed_val=0.3, pad_value=-7., dolinar_layers=2):
        '''
        dolinar_layers= number of photodetections
        pad_value: value not considered by the lstm
        valreg: regularisation value
        seed_val: interval of random parameter inizialitaion.
        '''
        super(Critic,self).__init__()

        self.pad_value = pad_value
        self.nature = nature
        self.dolinar_layers = dolinar_layers
        self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                  input_shape=(self.dolinar_layers, 2)) #(beta1, pad), (n1, beta2), (n2, guess). In general i will have (layer+1)
        self.lstm = tf.keras.layers.LSTM(500, return_sequences=True)

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



    def update_target_parameters(self,primary_net, tau=0.01):
        #### only
        # for i,j in zip(self.get_weights(), primary_net.get_weights()):
        #     tf.assign(i, tau*j + (i-tau)*i )
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
        # feat = tf.nn.dropout(feat, rate=0.01)
        feat = tf.nn.relu(self.l1(feat))
        # feat = tf.nn.dropout(feat, rate=0.01)
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.sigmoid(self.l4(feat))
        return feat


    def process_sequence(self,sample_buffer):
        """"
        sample_buffer: array of shape (N,2*self.layers +1), N>1

        gets data obtained from N experiments: data.shape = (N, 2L+1),
        where +1 accounts for the guess and 2L for (beta, outcome).

        [[a0, o1, a1, o2, a2, o3, a4]
         [same but other experiment]
        ]

        and returns an array of shape (experiments, self.layers, 2 ), as accepted by an RNN
        """
        batch_size = sample_buffer.shape[0]
        data = sample_buffer[:,0:(self.dolinar_layers+1+1)]
        padded_data = np.ones((batch_size,self.dolinar_layers+1, 2))*self.pad_value
        padded_data[:,0][:,0] = data[:,0]
        for k in range(1,LAYERS+1):
            padded_data[:,k] = data[:,[k,k+1]]

        rewards_obtained = np.zeros((batch_size, self.dolinar_layers+1))
        rewards_obtained[:,-1] = sample_buffer[:,-1]
        return padded_data, rewards_obtained


    def pad_single_sequence(self, seq, LAYERS=1):
        """"
        input: [a0, o1, a1, o2, a2, o3, a4]

        output: [[a0, pad], [o1, a1], [...]]

        the cool thing is that then you can put this to predict the greedy guess/action.
        """
        padded_data = np.ones((1,LAYERS+1, 2))*self.pad_value
        padded_data[0][0][0] = seq[0]
        #padded_data[0][0] = data[0]
        for k in range(1,LAYERS+1):
            padded_data[0][k] = seq[k:(k+2)]
        return padded_data

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
        for k in range(0,self.dolinar_layers-1):
            print(k)
            ll[:,k] = np.squeeze(self(b))[:,k+1] + ll[:,k]

        preds1 = self(b)
        b[:,-1][:,-1] = -b[:,1][:,1]
        preds2 = self(b)
        both = tf.concat([preds1,preds2],2)
        maxs = np.squeeze(tf.math.reduce_max(both,axis=2).numpy())
        ll[:,-2] = maxs[:,1] # This is the last befre the guess.. so the label is max_g Q(h-L, g)
        ll = np.expand_dims(ll,axis=1)
        return ll


    def give_favourite_guess(self,sequence_with_plus):
        """"
            important !! the 1!
        sequence should be [[beta, pad], [outcome, 1]] """
        pred_1 = self(sequence_with_plus)
        sequence_with_plus[:,1][:,1] = -sequence_with_plus[:,1][:,1]
        pred_2 = self(sequence_with_plus)
        both = tf.concat([pred_1,pred_2],2)
        maxs = np.squeeze(tf.argmax(both,axis=2).numpy())[1]

        guess = (-1)**maxs
        return  guess




##### ACTOR CLASSS ####
class Actor(tf.keras.Model):
    def __init__(self, nature, valreg=0.01, seed_val=0.1, pad_value = -7.,
                 dolinar_layers=2):
        super(Actor,self).__init__()
        self.dolinar_layers = dolinar_layers
        self.pad_value = pad_value
        self.nature = nature

        if nature == "primary":
            self.lstm = tf.keras.layers.LSTM(500, return_sequences=True, stateful=True)
            self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                  input_shape=(1, 1))
        elif nature == "target":
            self.lstm = tf.keras.layers.LSTM(500, return_sequences=True, stateful=False)
            self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                  input_shape=(self.dolinar_layers, 1)) #'cause i feed altoghether.
        else:
            print("Hey! the character is either primary or target")
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



    def update_target_parameters(self,primary_net, tau=0.01):
        #### only
        # for i,j in zip(self.get_weights(), primary_net.get_weights()):
        #     tf.assign(i, tau*j + (i-tau)*i )
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
        # feat = tf.nn.dropout(feat, rate=0.01)
        feat = tf.nn.relu(self.l1(feat))
        # feat = tf.nn.dropout(feat, rate=0.01)
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.sigmoid(self.l4(feat))

        return feat

    def targeted_sequence(self, experiences):

        #This function takes a vector of experiences:
        #vector = (\beta1, o1, \beta2, o2, \beta3, o3,...,o_L, guess)
        #and retrieves
        #(\beta1, o1, \beta2_target, o2, \beta3_target, o3, \beta4_target,... ,o_L, guess)
        #actor_target.lstm.reset_states() NO!
        if self.nature != "target":
            raise AttributeError("check the lstm memory of actor target, stateful == True ?")
            return
        export = experiences.copy()
        for index in range(1,2*self.dolinar_layers-1,2): # I consider from first outcome to last one (but guess)
            export[:,index+1] = np.squeeze(self(np.reshape(np.array(export[:,index]),
                                                                 (experiences.shape[0],1,1))))
        return export

    def __str__(self):
        return self.name

#
#
#
# def optimization_step(networks, optimizers, losses, buffer, batch_size=500., tau=0.01, repetitions=1):
#     actor_q0, critic_q0, critic_guess, target_guess = networks
#     optimizer_critic_guess,  optimizer_actor_l0, optimizer_critic_l0 = optimizers
#     train_loss_l0, train_loss_l1 = losses
#     for thoughts in range(repetitions):
#         experiences = buffer.sample(batch_size)
#
#         ##### update the critic guess according to rewards obtained
#         with tf.GradientTape() as tape:
#             tape.watch(critic_guess.trainable_variables)
#             preds_cguess = critic_guess(experiences[:,[0,1,2]])
#             labels_cguess = np.expand_dims(experiences[:,3],axis=1)
#             loss_prim_guess = tf.keras.losses.MSE(labels_cguess, preds_cguess)
#             loss_prim_guess = tf.reduce_mean(loss_prim_guess)
#             grads = tape.gradient(loss_prim_guess, critic_guess.trainable_variables)
#             optimizer_critic_guess.apply_gradients(zip(grads, critic_guess.trainable_variables))
#             train_loss_l1(loss_prim_guess)
#
#         ##### update the target guess ######
#         target_guess.update_target_parameters(critic_guess, tau=0.01) #check this value !
#
#         #### obtain the labels for the update of Q(\beta)
#         labels_critic_l0 = target_guess.calculate_greedy_from_batch(experiences[:,[0,1,2]]) #greedy from target; this is the label for net_0!!
#
#         with tf.GradientTape() as tape:
#             tape.watch(critic_q0.trainable_variables)
#             preds0 = critic_q0(np.expand_dims(experiences[:,0],axis=1))
#             loss_0 = tf.keras.losses.MSE(labels_critic_l0,preds0)
#             loss_0 = tf.reduce_mean(loss_0)
#             grads0 = tape.gradient(loss_0, critic_q0.trainable_variables)
#             optimizer_critic_l0.apply_gradients(zip(grads0, critic_q0.trainable_variables))
#         train_loss_l0(loss_0)
#
#         #### obtain the components for the chain for the update of \pi( h_0 = nada!) = \beta
#         with tf.GradientTape() as tape:
#             actions = actor_q0(np.expand_dims(np.zeros(len(experiences)),axis=1))
#             tape.watch(actions)
#             qvals = critic_q0(actions)
#             dq_da = tape.gradient(qvals, actions)
#
#         ### update actor \pi( h_0) = \beta
#         with tf.GradientTape() as tape:
#             actions = actor_q0(np.expand_dims(np.zeros(len(experiences)),axis=1))
#             da_dtheta = tape.gradient(actions, actor_q0.trainable_variables, output_gradients=-dq_da)
#             optimizer_actor_l0.apply_gradients(zip(da_dtheta, actor_q0.trainable_variables))
#     return
