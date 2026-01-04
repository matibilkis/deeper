"""
Actor network implementations for DDPG.

The Actor network outputs continuous actions (policy) for the quantum receiver optimization.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

tf.keras.backend.set_floatx('float32')


class Actor(tf.keras.Model):
    """
    Actor network for DDPG that outputs continuous actions.
    
    The network takes state information and outputs a continuous action value
    (e.g., displacement amplitude beta for quantum receiver).
    
    Args:
        input_dim: Input dimension (1 for layer 0, 3 for layer 2 in Kennedy receiver)
        valreg: Regularization value for L1/L2 regularization (default: 0.01)
        seed_val: Range for uniform weight initialization (default: 0.1)
    """
    
    def __init__(self, input_dim=1, valreg=0.01, seed_val=0.1):
        super(Actor, self).__init__()

        self.l1 = Dense(
            50, 
            input_shape=(input_dim,),
            kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            bias_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            kernel_regularizer=tf.keras.regularizers.l1(valreg),
            activity_regularizer=tf.keras.regularizers.l2(valreg)
        )

        self.l2 = Dense(
            50,
            kernel_regularizer=tf.keras.regularizers.l1(valreg),
            activity_regularizer=tf.keras.regularizers.l2(valreg),
            kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            bias_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val)
        )
        
        self.l3 = Dense(
            50,
            kernel_regularizer=tf.keras.regularizers.l1(valreg),
            activity_regularizer=tf.keras.regularizers.l2(valreg),
            kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            bias_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val)
        )
        
        self.l4 = Dense(
            50,
            kernel_regularizer=tf.keras.regularizers.l1(valreg),
            activity_regularizer=tf.keras.regularizers.l2(valreg),
            kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            bias_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val)
        )
        
        self.l5 = Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l1(valreg),
            activity_regularizer=tf.keras.regularizers.l2(valreg),
            kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            bias_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val)
        )

    def update_target_parameters(self, primary_net, tau=0.01):
        """
        Soft update of target network parameters.
        
        Updates this network's weights as: 
        weights = tau * primary_weights + (1 - tau) * target_weights
        
        Args:
            primary_net: The primary network to copy weights from
            tau: Soft update coefficient (default: 0.01)
        """
        prim_weights = primary_net.get_weights()
        targ_weights = self.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(tau * prim_weights[i] + (1 - tau) * targ_weights[i])
        self.set_weights(weights)
        return

    def call(self, input):
        """
        Forward pass through the actor network.
        
        Args:
            input: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Continuous action value (tanh output, bounded)
        """
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l4(feat))
        feat = tf.nn.tanh(self.l5(feat))
        return feat

    def __str__(self):
        return self.name

