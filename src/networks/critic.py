"""
Critic network implementations for DDPG.

The Critic network estimates Q-values for state-action pairs.
Supports both feedforward and recurrent (LSTM) architectures.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np

tf.keras.backend.set_floatx('float32')
tf.compat.v1.enable_eager_execution()


class Critic(tf.keras.Model):
    """
    Critic network with LSTM for sequence processing.
    
    This network processes sequences of (beta, outcome, guess) tuples
    using LSTM layers to handle temporal dependencies in quantum receiver
    decision-making.
    
    Args:
        valreg: Regularization value for L1/L2 regularization (default: 0.01)
        seed_val: Range for uniform weight initialization (default: 0.3)
        pad_value: Padding value for masking sequences (default: -7.0)
    """
    
    def __init__(self, valreg=0.01, seed_val=0.3, pad_value=-7.):
        super(Critic, self).__init__()

        self.pad_value = pad_value
        self.mask = tf.keras.layers.Masking(
            mask_value=pad_value,
            input_shape=(2, 2)
        )
        self.lstm = tf.keras.layers.LSTM(500, return_sequences=True)

        self.l1 = Dense(
            250,
            kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            bias_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            kernel_regularizer=tf.keras.regularizers.l1(valreg),
            activity_regularizer=tf.keras.regularizers.l2(valreg)
        )

        self.l2 = Dense(
            100,
            kernel_regularizer=tf.keras.regularizers.l1(valreg),
            activity_regularizer=tf.keras.regularizers.l2(valreg),
            kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            bias_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val)
        )

        self.l3 = Dense(
            100,
            kernel_regularizer=tf.keras.regularizers.l1(valreg),
            activity_regularizer=tf.keras.regularizers.l2(valreg),
            kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            bias_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val)
        )

        self.l4 = Dense(
            1,
            kernel_regularizer=tf.keras.regularizers.l1(valreg),
            activity_regularizer=tf.keras.regularizers.l2(valreg),
            kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
            bias_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val)
        )

    def update_target_parameters(self, primary_net, tau=0.01):
        """
        Soft update of target network parameters.
        
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

    def call(self, inputs):
        """
        Forward pass through the critic network.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, 2)
                   Sequences of [beta, outcome/guess] pairs
                   
        Returns:
            Q-value predictions
        """
        feat = self.mask(inputs)
        feat = self.lstm(feat)
        feat = tf.nn.relu(self.l1(feat))
        feat = tf.nn.relu(self.l2(feat))
        feat = tf.nn.relu(self.l3(feat))
        feat = tf.nn.sigmoid(self.l4(feat))
        return feat

    def process_sequence(self, sample_buffer, LAYERS=1):
        """
        Process batch of experiences into RNN-compatible format.
        
        Converts data from shape (N, 2L+1) to (N, L+1, 2) for RNN input.
        The +1 accounts for the guess, and 2L for (beta, outcome) pairs.
        
        Args:
            sample_buffer: Array of shape (batch_size, 2L+1) containing
                         [beta, outcome1, beta1, outcome2, ..., guess]
            LAYERS: Number of layers in the receiver (default: 1)
            
        Returns:
            padded_data: Array of shape (batch_size, LAYERS+1, 2)
            rewards_obtained: Array of shape (batch_size, LAYERS+1) with rewards
        """
        batch_size = sample_buffer.shape[0]
        data = sample_buffer[:, 0:(LAYERS+1+1)]
        padded_data = np.ones((batch_size, LAYERS+1, 2)) * self.pad_value
        padded_data[:, 0][:, 0] = data[:, 0]
        for k in range(1, LAYERS+1):
            padded_data[:, k] = data[:, [k, k+1]]

        rewards_obtained = np.zeros((batch_size, LAYERS+1)).astype(np.float32)
        rewards_obtained[:, -1] = sample_buffer[:, -1]
        return padded_data, rewards_obtained

    def pad_single_sequence(self, seq, LAYERS=1):
        """
        Pad a single sequence for prediction.
        
        Args:
            seq: Sequence [a0, o1, a1, o2, a2, o3, a4]
            LAYERS: Number of layers
            
        Returns:
            Padded sequence of shape (1, LAYERS+1, 2)
        """
        padded_data = np.ones((1, LAYERS+1, 2)) * self.pad_value
        padded_data[0][0][0] = seq[0]
        for k in range(1, LAYERS+1):
            padded_data[0][k] = seq[k:(k+2)]
        return padded_data

    def give_td_error_Kennedy_guess(self, batched_input, sequential_rews_with_zeros):
        """
        Calculate TD error labels for Kennedy receiver guess network.
        
        Computes max Q-values over possible guesses to create target labels.
        
        Args:
            batched_input: Batched input sequences
            sequential_rews_with_zeros: Sequential rewards with zeros
            
        Returns:
            TD error labels for training
        """
        b = batched_input.copy()
        ll = sequential_rews_with_zeros.copy()
        preds1 = self(b)
        b[:, 1][:, 1] = -b[:, 1][:, 1]
        preds2 = self(b)
        both = tf.concat([preds1, preds2], 2)
        maxs = np.squeeze(tf.math.reduce_max(both, axis=2).numpy())
        ll[:, 0] = maxs[:, 1] + ll[:, 0]
        ll = np.expand_dims(ll, axis=1)
        return ll

    def give_favourite_guess(self, sequence_with_plus):
        """
        Get the greedy guess (action) for a given sequence.
        
        Important: sequence should be [[beta, pad], [outcome, 1]]
        
        Args:
            sequence_with_plus: Input sequence with positive guess
            
        Returns:
            Optimal guess value (-1 or 1)
        """
        pred_1 = self(sequence_with_plus)
        sequence_with_plus[:, 1][:, 1] = -sequence_with_plus[:, 1][:, 1]
        pred_2 = self(sequence_with_plus)
        both = tf.concat([pred_1, pred_2], 2)
        maxs = np.squeeze(tf.argmax(both, axis=2).numpy())[1]
        guess = (-1)**maxs
        return guess


class CriticFeedforward(tf.keras.Model):
    """
    Feedforward Critic network (non-recurrent variant).
    
    Simpler critic network without LSTM, for comparison or simpler problems.
    
    Args:
        input_dim: Input dimension
        valreg: Regularization value (default: 0.01)
        seed_val: Range for uniform weight initialization (default: 0.1)
    """
    
    def __init__(self, input_dim, valreg=0.01, seed_val=0.1):
        super(CriticFeedforward, self).__init__()

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
        """Soft update of target network parameters."""
        prim_weights = primary_net.get_weights()
        targ_weights = self.get_weights()
        weights = []
        for i in tf.range(len(prim_weights)):
            weights.append(tau * prim_weights[i] + (1 - tau) * targ_weights[i])
        self.set_weights(weights)
        return

    def call(self, input):
        """Forward pass through feedforward critic."""
        feat = tf.nn.relu(self.l1(input))
        feat = tf.nn.relu(self.l4(feat))
        feat = tf.nn.sigmoid(self.l5(feat))
        return feat

    def calculate_greedy_from_batch(self, batch):
        """
        Calculate greedy action from batch of experiences.
        
        Args:
            batch: Array of shape (batch_size, 3) with [beta, n, guess]
            
        Returns:
            Max Q-values over possible guesses
        """
        a = batch.copy()
        preds1 = self(a)
        a[:, 2] = -a[:, 2]
        preds2 = self(a)
        both = tf.concat([preds1, preds2], 1)
        maxs = np.squeeze(tf.math.reduce_max(both, axis=1))
        maxs = np.expand_dims(maxs, axis=1)
        return maxs

    def give_favourite_guess(self, beta, outcome):
        """
        Get the greedy guess for given beta and outcome.
        
        Args:
            beta: Displacement parameter
            outcome: Measurement outcome (0 or 1)
            
        Returns:
            Optimal guess value (-1 or 1)
        """
        h1a2 = np.array([[beta, outcome, -1.]])
        pred_minus = self(h1a2)
        h1a2[:, 2] = 1.
        pred_plus = self(h1a2)
        both = tf.concat([pred_plus, pred_minus], 1)
        maxs = tf.argmax(both, axis=1)
        guess = (-1)**maxs.numpy()[0]
        return guess

