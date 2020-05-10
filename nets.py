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
    activity_regularizer=tf.keras.regularizers.l2(valreg))

        self.l2 = Dense(50, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l3 = Dense(1, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))




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
        feat = tf.nn.relu(self.l1(feat))
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




    def give_favourite_guess(self,sequence):
        """"sequence should be [[beta, pad], [outcome, guess]] """
        pred_1 = self(sequence)
        sequence[:,1][:,1] = -sequence[:,1][:,1]
        pred_2 = self(sequence)
        both = tf.concat([pred_1,pred_2],1)
        maxs = tf.argmax(both,axis=1)
        guess = (-1)**maxs.numpy()[0][0]
        return guess



##### ACTOR CLASSS ####
class Actor(tf.keras.Model):
    #input_dim: 1 if layer=0, 3 if layer= 2, for the Kennedy receiver ##
    def __init__(self, input_dim=1, valreg=0.01, seed_val=0.1):
        super(Actor,self).__init__()

        self.l1 = Dense(50, input_shape=(input_dim,),kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
        kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg))

        self.l2 = Dense(50, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))
        self.l3 = Dense(50, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l4 = Dense(50, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))

        self.l5 = Dense(1, kernel_regularizer=tf.keras.regularizers.l1(valreg),
    activity_regularizer=tf.keras.regularizers.l2(valreg),
    kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
    bias_initializer = tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val))


    def update_target_parameters(self,primary_net, tau=0.01):
        #### only
        prim_weights = primary_net.get_weights()
        targ_weights = self.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(tau * prim_weights[i] + (1 - tau) * targ_weights[i])
        self.set_weights(weights)
        return

    def call(self, input):
        feat = tf.nn.relu(self.l1(input))
#        feat = tf.nn.dropout(feat, rate=0.01)
 #       feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.dropout(feat, rate=0.01)
   #     feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.relu(self.l4(feat))
        feat = tf.nn.sigmoid(self.l5(feat))
        return feat


    def __str__(self):
        return self.name



def optimization_step(networks, optimizers, losses, buffer, batch_size=500., tau=0.01, repetitions=1):
    actor_q0, critic_q0, critic_guess, target_guess = networks
    optimizer_critic_guess,  optimizer_actor_l0, optimizer_critic_l0 = optimizers
    train_loss_l0, train_loss_l1 = losses
    for thoughts in range(repetitions):
        experiences = buffer.sample(batch_size)

        ##### update the critic guess according to rewards obtained
        with tf.GradientTape() as tape:
            tape.watch(critic_guess.trainable_variables)
            preds_cguess = critic_guess(experiences[:,[0,1,2]])
            labels_cguess = np.expand_dims(experiences[:,3],axis=1)
            loss_prim_guess = tf.keras.losses.MSE(labels_cguess, preds_cguess)
            loss_prim_guess = tf.reduce_mean(loss_prim_guess)
            grads = tape.gradient(loss_prim_guess, critic_guess.trainable_variables)
            optimizer_critic_guess.apply_gradients(zip(grads, critic_guess.trainable_variables))
            train_loss_l1(loss_prim_guess)

        ##### update the target guess ######
        target_guess.update_target_parameters(critic_guess, tau=0.01) #check this value !

        #### obtain the labels for the update of Q(\beta)
        labels_critic_l0 = target_guess.calculate_greedy_from_batch(experiences[:,[0,1,2]]) #greedy from target; this is the label for net_0!!

        with tf.GradientTape() as tape:
            tape.watch(critic_q0.trainable_variables)
            preds0 = critic_q0(np.expand_dims(experiences[:,0],axis=1))
            loss_0 = tf.keras.losses.MSE(labels_critic_l0,preds0)
            loss_0 = tf.reduce_mean(loss_0)
            grads0 = tape.gradient(loss_0, critic_q0.trainable_variables)
            optimizer_critic_l0.apply_gradients(zip(grads0, critic_q0.trainable_variables))
        train_loss_l0(loss_0)

        #### obtain the components for the chain for the update of \pi( h_0 = nada!) = \beta
        with tf.GradientTape() as tape:
            actions = actor_q0(np.expand_dims(np.zeros(len(experiences)),axis=1))
            tape.watch(actions)
            qvals = critic_q0(actions)
            dq_da = tape.gradient(qvals, actions)

        ### update actor \pi( h_0) = \beta
        with tf.GradientTape() as tape:
            actions = actor_q0(np.expand_dims(np.zeros(len(experiences)),axis=1))
            da_dtheta = tape.gradient(actions, actor_q0.trainable_variables, output_gradients=-dq_da)
            optimizer_actor_l0.apply_gradients(zip(da_dtheta, actor_q0.trainable_variables))
    return
