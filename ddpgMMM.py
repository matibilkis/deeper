from nets import Critic, Actor
import numpy as np
import tensorflow as tf
import numpy as np


class DDPG():
    def __init__(self,name="DDPG", actor_units=[3,4], critic_units=[3,4], tau=0.005,
                 memory_capacity=int(1e6), lr_actor=0.001, lr_critic=0.001, max_grad=10):

        self.memory_capacity = memory_capacity
        self.tau = tau
        self.device = "/cpu:0" #i put it just in case
        self.max_grad = max_grad
        self.max_value_action = 2

        ### Define the 4 NN's ####
        self.actor = Actor(units = actor_units, state_shape=(2,))
        self.actor_target = Actor(units = actor_units, state_shape=(2,))
        self.critic = Critic(units=critic_units, state_shape = (2,))
        self.critic_target = Critic(units=critic_units, state_shape = (2,))
        ### Define the 4 NN's ####

        #### we'd need to do this soft update in the targets...###
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer =  tf.keras.optimizers.Adam(learning_rate=lr_critic)

    def get_action(self, state):
        ##
        state = np.expand_dims(state,axis=0).astype(np.float32)
        #example state = np.array([0]) ---> array([[0.]], dtype=float32)
        action = self._get_action_body(tf.constant(state), sigma, self.max_value_action ) #this gives the action with some noise
        return action

    @tf.function
    def _get_action_body(self, state, sigma, max_action):
        #gives an action with some noise
        with tf.device(self.device):
            action = self.actor(state)
            action += tf.random.normal(shape=action.shape,
                                      mean=0., stdev=sigma, dtype=tf.float32)
            return tf.clip_by_value(action, -max_action, max_action)


    def train(self, states, actions, next_states, rewards, done, weights=None):
        if weights is None:
            weights = np.ones_like(rewards)
        actor_loss, critic_loss, td_erros = self._train_body(states, actions, next_states, rewards, done, weights)


        if actor_loss is not None:
            tf.summary.scalar(name=self.policy_name+"/actor_loss",
                              data=actor_loss)
        tf.summary.scalar(name=self.policy_name+"/critic_loss",
                          data=critic_loss)

        return td_errors

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, weights):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                td_errors = self._compute_td_error_body(states, actions, next_states, rewards, done)
                critic_loss = tf.reduce_mean( self.huber_loss(td_errors, delta = self.max_grad) ) #it's like np.mean(...) and set a maximum to the value of the gradient. See https://en.wikipedia.org/wiki/Huber_loss

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                next_action = self.actor(states)
                actor_loss = -tf.recude_mean(self.critic([states,next_action])) #Cool, this is < Q[s, a]>

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients( zip(actor_grad, self.actor.trainable_variables) )

            ##### update target networks here ####
            self.update_target_variables()
            ##### update target networks here ####
        return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards,done):
        if isinstance(actions, tf.Tensor):
            rewards = tf.expand_dims(rewards, axis=1)
            dones = tf.expand_dims(dones, 1)
        td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones)
        return np.ravel(td_errors.numpy())

    @tf.function
    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):
        with tf.device(self.device):
            not_dones = 1. -dones
            target_Q = self.critic_target([next_states, self.actor_target(next_states)])

            target_Q = rewards + (not_dones*self.discount*target_Q)
            target_Q = tf.stop_gradient(target_Q) #makes constant ?
            current_Q = self.critic([states, actions])
            td_errors = target_Q - current_Q
        return td_errors

    def update_target_variables(self):
        for i,k in enumerate(self.critic_target.trainable_variables):
            k.assign(tf.multiply(self.critic.trainable_variables[i], self.tau) + tf.multiply(i,1-self.tau))

        for i,k in enumerate(self.actor_target.trainable_variables):
            k.assign(tf.multiply(self.actor.trainable_variables[i], self.tau) + tf.multiply(i,1-self.tau))
        return

    def huber_loss(x, delta=1.):
        """"Compute the huber loss
            https://en.wikipedia.org/wiki/Huber_loss
        """
        delta = tf.ones_like(x)*delta
        less_than_max = 0.5*tf.square(x)
        greater_than_max = delta*(tf.abs(x) - 0.5*delta)
        return tf.where( tf.abs(x) <= delta, x=less_than_max, y=greater_than_max) #condition, case1, case2



                #####
