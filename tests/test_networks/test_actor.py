"""
Unit tests for Actor network.
"""

import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.networks.actor import Actor


class TestActor(unittest.TestCase):
    """Test cases for Actor network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.actor = Actor(input_dim=1, valreg=0.01, seed_val=0.1)
        # Initialize network
        self.actor(np.array([[0.]]).astype(np.float32))
    
    def test_actor_initialization(self):
        """Test that Actor can be initialized."""
        self.assertIsNotNone(self.actor)
        self.assertEqual(len(self.actor.layers), 5)
    
    def test_actor_forward_pass(self):
        """Test forward pass through Actor network."""
        input_tensor = np.array([[0.], [0.5], [-0.5]]).astype(np.float32)
        output = self.actor(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (3, 1))
        
        # Check output is bounded (tanh outputs in [-1, 1])
        self.assertTrue(np.all(output.numpy() >= -1.0))
        self.assertTrue(np.all(output.numpy() <= 1.0))
    
    def test_actor_single_input(self):
        """Test Actor with single input."""
        input_tensor = np.array([[0.]]).astype(np.float32)
        output = self.actor(input_tensor)
        
        self.assertEqual(output.shape, (1, 1))
        self.assertIsInstance(output.numpy()[0][0], (float, np.floating))
    
    def test_actor_batch_processing(self):
        """Test Actor processes batches correctly."""
        batch_size = 10
        input_batch = np.random.randn(batch_size, 1).astype(np.float32)
        output = self.actor(input_batch)
        
        self.assertEqual(output.shape, (batch_size, 1))
    
    def test_target_network_update(self):
        """Test soft update of target network parameters."""
        primary_actor = Actor(input_dim=1)
        target_actor = Actor(input_dim=1)
        
        # Initialize both
        primary_actor(np.array([[0.]]).astype(np.float32))
        target_actor(np.array([[0.]]).astype(np.float32))
        
        # Get initial weights
        initial_weights = [w.copy() for w in target_actor.get_weights()]
        
        # Update target
        tau = 0.1
        target_actor.update_target_parameters(primary_actor, tau=tau)
        
        # Check weights changed
        updated_weights = target_actor.get_weights()
        weights_changed = False
        for init_w, upd_w in zip(initial_weights, updated_weights):
            if not np.allclose(init_w, upd_w):
                weights_changed = True
                break
        
        self.assertTrue(weights_changed, "Target network weights should be updated")


if __name__ == '__main__':
    unittest.main()

