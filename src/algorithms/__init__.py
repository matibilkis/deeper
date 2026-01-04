"""
DDPG algorithm implementations.

Contains the main DDPG training algorithm and replay buffer.
"""

from src.algorithms.ddpg import ddpg_kennedy, optimization_step
from src.algorithms.buffer import ReplayBuffer

__all__ = ['ddpg_kennedy', 'optimization_step', 'ReplayBuffer']

