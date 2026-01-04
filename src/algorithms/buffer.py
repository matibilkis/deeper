"""
Experience replay buffer for DDPG.

Implements a circular buffer for storing and sampling experiences
for off-policy reinforcement learning.
"""

import numpy as np
from collections import deque
import random
try:
    from src.quantum.receivers import Prob, ps_maxlik, qval
except ImportError:
    # For relative imports when running as module
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.quantum.receivers import Prob, ps_maxlik, qval


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling experiences.
    
    Stores tuples of (beta, outcome, guess, reward) and provides
    batch sampling for training. Also creates test datasets for evaluation.
    
    Args:
        buffer_size: Maximum size of the replay buffer (default: 10^3)
    """
    
    def __init__(self, buffer_size=10**3):
        self.buffer_size = buffer_size
        self.count = 0
        self.betas = np.arange(-1.5, 1.5, 0.01)
        self.buffer = deque()
        self.create_test_datasets()

    def create_test_datasets(self):
        """
        Create test datasets for evaluation.
        
        Generates test data for:
        - Layer 0: (beta, ps_maxlik(beta)) pairs
        - Layer 1: (beta, n, g, qval(beta, n, g)) tuples
        """
        dt_l0 = []
        dt_l1 = []

        for k in self.betas:
            dt_l0.append([k, ps_maxlik(k)])
            for n in [0., 1.]:
                for g in [-1., 1.]:
                    dt_l1.append([k, n, g, qval(k, n, g)])
        self.test_l0 = np.array(dt_l0)
        self.test_l1 = np.array(dt_l1)
        return

    def add(self, beta, outcome, guess, reward):
        """
        Add an experience to the buffer.
        
        Args:
            beta: Displacement parameter
            outcome: Measurement outcome (0 or 1)
            guess: Agent's guess (-1 or 1)
            reward: Reward received
        """
        experience = (beta, outcome, guess, reward)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """Return current size of the buffer."""
        return self.count

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Array of shape (batch_size, 4) with columns [beta, outcome, guess, reward]
        """
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, int(batch_size))
        beta_batch, outcome_batch, guess_batch, r_batch = list(
            map(np.array, list(zip(*batch))
        )
        return np.array([beta_batch, outcome_batch, guess_batch, r_batch]).transpose().astype(np.float32)

    def clear(self):
        """Clear the buffer and reset count."""
        self.buffer.clear()
        self.count = 0

