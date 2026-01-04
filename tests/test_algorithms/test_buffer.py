"""
Unit tests for ReplayBuffer.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.algorithms.buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(buffer_size=100)
    
    def test_buffer_initialization(self):
        """Test that buffer can be initialized."""
        self.assertIsNotNone(self.buffer)
        self.assertEqual(self.buffer.buffer_size, 100)
        self.assertEqual(self.buffer.count, 0)
        self.assertIsNotNone(self.buffer.test_l0)
        self.assertIsNotNone(self.buffer.test_l1)
    
    def test_add_experience(self):
        """Test adding experiences to buffer."""
        initial_count = self.buffer.count
        
        self.buffer.add(beta=0.5, outcome=0.0, guess=1.0, reward=1.0)
        
        self.assertEqual(self.buffer.count, initial_count + 1)
        self.assertEqual(self.buffer.size(), initial_count + 1)
    
    def test_buffer_filling(self):
        """Test buffer fills up correctly."""
        # Add experiences up to buffer size
        for i in range(50):
            self.buffer.add(beta=0.5, outcome=0.0, guess=1.0, reward=1.0)
        
        self.assertEqual(self.buffer.count, 50)
        self.assertEqual(self.buffer.size(), 50)
    
    def test_buffer_overflow(self):
        """Test buffer handles overflow correctly (FIFO)."""
        buffer_size = 10
        
        # Fill buffer beyond capacity
        for i in range(15):
            self.buffer.add(beta=float(i), outcome=0.0, guess=1.0, reward=1.0)
        
        # Count should not exceed buffer size
        self.assertLessEqual(self.buffer.count, buffer_size)
        self.assertEqual(self.buffer.count, buffer_size)
    
    def test_sample_batch(self):
        """Test sampling batches from buffer."""
        # Add some experiences
        for i in range(20):
            self.buffer.add(beta=0.5, outcome=0.0, guess=1.0, reward=1.0)
        
        # Sample a batch
        batch_size = 5
        batch = self.buffer.sample(batch_size)
        
        # Check batch shape and content
        self.assertEqual(batch.shape, (batch_size, 4))
        self.assertEqual(batch.dtype, np.float32)
        
        # Check columns: [beta, outcome, guess, reward]
        self.assertTrue(np.all(batch[:, 0] >= -1.5))  # beta range
        self.assertTrue(np.all(batch[:, 0] <= 1.5))
        self.assertTrue(np.all(np.isin(batch[:, 1], [0.0, 1.0])))  # outcome
        self.assertTrue(np.all(np.isin(batch[:, 2], [-1.0, 1.0])))  # guess
        self.assertTrue(np.all(np.isin(batch[:, 3], [0.0, 1.0])))  # reward
    
    def test_sample_smaller_than_buffer(self):
        """Test sampling when buffer has fewer experiences than batch size."""
        # Add only 3 experiences
        for i in range(3):
            self.buffer.add(beta=0.5, outcome=0.0, guess=1.0, reward=1.0)
        
        # Try to sample more
        batch = self.buffer.sample(10)
        
        # Should return all available (3)
        self.assertEqual(batch.shape[0], 3)
    
    def test_clear_buffer(self):
        """Test clearing the buffer."""
        # Add some experiences
        for i in range(10):
            self.buffer.add(beta=0.5, outcome=0.0, guess=1.0, reward=1.0)
        
        self.assertGreater(self.buffer.count, 0)
        
        # Clear buffer
        self.buffer.clear()
        
        self.assertEqual(self.buffer.count, 0)
        self.assertEqual(self.buffer.size(), 0)
    
    def test_test_datasets(self):
        """Test that test datasets are created correctly."""
        # Check test_l0 shape: (num_betas, 2) -> [beta, ps_maxlik(beta)]
        self.assertEqual(self.buffer.test_l0.shape[1], 2)
        
        # Check test_l1 shape: (num_betas * 2 * 2, 4) -> [beta, n, g, qval]
        # For each beta, 2 outcomes, 2 guesses
        self.assertEqual(self.buffer.test_l1.shape[1], 4)
        
        # Check value ranges
        self.assertTrue(np.all(self.buffer.test_l0[:, 0] >= -1.5))  # beta range
        self.assertTrue(np.all(self.buffer.test_l0[:, 0] <= 1.5))
        self.assertTrue(np.all(self.buffer.test_l0[:, 1] >= 0.0))  # probability
        self.assertTrue(np.all(self.buffer.test_l0[:, 1] <= 1.0))


if __name__ == '__main__':
    unittest.main()

