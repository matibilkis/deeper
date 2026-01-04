"""
Unit tests for Critic network.
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.networks.critic import Critic, CriticFeedforward


class TestCritic(unittest.TestCase):
    """Test cases for Critic network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.critic = Critic(valreg=0.01, seed_val=0.3, pad_value=-7.0)
        self.critic_ff = CriticFeedforward(input_dim=3, valreg=0.01, seed_val=0.1)
    
    def test_critic_initialization(self):
        """Test that Critic can be initialized."""
        self.assertIsNotNone(self.critic)
        self.assertEqual(self.critic.pad_value, -7.0)
    
    def test_critic_forward_pass(self):
        """Test forward pass through Critic network."""
        # Create a batch of sequences: (batch_size, sequence_length, 2)
        batch_size = 5
        sequence_length = 2
        input_sequences = np.random.randn(batch_size, sequence_length, 2).astype(np.float32)
        
        output = self.critic(input_sequences)
        
        # Check output shape: should be (batch_size, sequence_length, 1) due to return_sequences=True
        self.assertEqual(output.shape, (batch_size, sequence_length, 1))
        
        # Check output is in valid range (sigmoid outputs in [0, 1])
        self.assertTrue(np.all(output.numpy() >= 0.0))
        self.assertTrue(np.all(output.numpy() <= 1.0))
    
    def test_process_sequence(self):
        """Test sequence processing for RNN input."""
        # Sample buffer format: (batch_size, 2L+1) where L=1
        # Format: [beta, outcome, guess]
        batch_size = 4
        sample_buffer = np.array([
            [0.5, 0.0, 1.0],   # beta, outcome, guess
            [-0.3, 1.0, -1.0],
            [0.8, 0.0, 1.0],
            [-0.2, 1.0, -1.0]
        ]).astype(np.float32)
        
        padded_data, rewards = self.critic.process_sequence(sample_buffer, LAYERS=1)
        
        # Check shapes
        self.assertEqual(padded_data.shape, (batch_size, 2, 2))  # (batch, L+1, 2)
        self.assertEqual(rewards.shape, (batch_size, 2))  # (batch, L+1)
        
        # Check padding
        self.assertEqual(padded_data[0, 0, 0], sample_buffer[0, 0])  # beta should be first element
        self.assertEqual(padded_data[0, 0, 1], self.critic.pad_value)  # pad value
    
    def test_pad_single_sequence(self):
        """Test padding of single sequence."""
        seq = [0.5, 0.0, 1.0]  # beta, outcome, guess
        padded = self.critic.pad_single_sequence(seq, LAYERS=1)
        
        self.assertEqual(padded.shape, (1, 2, 2))
        self.assertEqual(padded[0, 0, 0], seq[0])  # beta
        self.assertEqual(padded[0, 0, 1], self.critic.pad_value)  # pad
    
    def test_target_network_update(self):
        """Test soft update of target network parameters."""
        primary_critic = Critic()
        target_critic = Critic()
        
        # Get initial weights
        initial_weights = [w.copy() for w in target_critic.get_weights()]
        
        # Update target
        tau = 0.1
        target_critic.update_target_parameters(primary_critic, tau=tau)
        
        # Check weights changed
        updated_weights = target_critic.get_weights()
        weights_changed = False
        for init_w, upd_w in zip(initial_weights, updated_weights):
            if not np.allclose(init_w, upd_w):
                weights_changed = True
                break
        
        self.assertTrue(weights_changed, "Target network weights should be updated")
    
    def test_critic_feedforward(self):
        """Test feedforward Critic variant."""
        input_tensor = np.array([[0.5, 0.0, 1.0], [-0.3, 1.0, -1.0]]).astype(np.float32)
        output = self.critic_ff(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 1))
        
        # Check output is in valid range
        self.assertTrue(np.all(output.numpy() >= 0.0))
        self.assertTrue(np.all(output.numpy() <= 1.0))
    
    def test_give_favourite_guess(self):
        """Test greedy guess selection."""
        # Create sequence: [[beta, pad], [outcome, 1]]
        sequence = np.array([[[0.5, -7.0], [0.0, 1.0]]]).astype(np.float32)
        guess = self.critic.give_favourite_guess(sequence)
        
        # Guess should be either -1 or 1
        self.assertIn(guess, [-1.0, 1.0])


if __name__ == '__main__':
    unittest.main()

