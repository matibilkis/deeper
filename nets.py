import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
tf.keras.backend.set_floatx('float64')


class Critic(tf.keras.Model):
    #input_dim: 1 if layer=0, 3 if layer= 2, for the Kennedy receiver ##
    def __init__(self, input_dim, valreg=0.01, seed_val=0.1):
        super(Critic,self).__init__()

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
  #      feat = tf.nn.dropout(feat, rate=0.01)
   #     feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.relu(self.l4(feat))
        feat = tf.nn.sigmoid(self.l5(feat))
        return feat

    def calculate_greedy_from_batch(self, batch):
        """ this function is only to intended for Q(n, beta, guess).
        Assuming batch = np.array([[beta, n, guess], [beta1, n1, guess], ...])

        """
        a = batch.copy()
        preds1 = self(a)
        a[:,2] = -a[:,2]
        preds2 = self(a)
        both = tf.concat([preds1,preds2],1)
        maxs = np.squeeze(tf.math.reduce_max(both,axis=1))
        maxs = np.expand_dims(maxs, axis=1)
        return maxs

    def give_favourite_guess(self, beta, outcome):
        """"This funciton is only intended for Q(n, beta, guess)"""
        h1a2 = np.array([[beta, outcome,-1.]])
        pred_minus = self(h1a2)
        h1a2[:,2] = 1.
        pred_plus = self(h1a2)
        both = tf.concat([pred_plus,pred_minus],1)
        maxs = tf.argmax(both,axis=1)
        guess = (-1)**maxs.numpy()[0]
        return guess

    def __str__(self):
        return self.name



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
        feat = tf.nn.tanh(self.l5(feat))
        return feat

    def calculate_greedy_from_batch(self, batch):
        """ this function is only to intended for Q(n, beta, guess)"""
        a = batch[1].copy()
        preds1 = self(a)
        a[:,2] = -a[:,2]
        preds2 = self(a)
        both = tf.concat([preds1,preds2],1)
        maxs = np.squeeze(tf.math.reduce_max(both,axis=1))
        maxs = np.expand_dims(maxs, axis=1)
        return maxs

    def __str__(self):
        return self.name




def testing_data(losses, buffer,networks):
    test_loss_l0, test_loss_l1 = losses
    actor_q0, critic_q0, critic_guess, target_guess = networks
    ### this is the test data for the guess network, defined in Dataset() classs
    predstest = critic_guess(buffer.test_l1[:,[0,1,2]])
    targets_1 = np.expand_dims(buffer.test_l1[:,3], axis=1)
    loss_test_l1 = tf.keras.losses.MSE(targets_1, predstest)
    loss_test_l1 = tf.reduce_mean(loss_test_l1)
    test_loss_l1(loss_test_l1)

    ### this is the test data for the \hat{Q}('beta) #####
    preds_test_l0 = critic_q0(np.expand_dims(buffer.test_l0[:,0], axis=1))
    loss_y0 = tf.keras.losses.MSE(np.expand_dims(buffer.test_l0[:,1], axis=1), preds_test_l0)
    loss_y0 = tf.reduce_mean(loss_y0)
    test_loss_l0(loss_y0)

    return



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
