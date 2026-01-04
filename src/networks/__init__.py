"""
Neural network implementations for DDPG.

Contains Actor and Critic network classes.
"""

from src.networks.actor import Actor
from src.networks.critic import Critic, CriticFeedforward

__all__ = ['Actor', 'Critic', 'CriticFeedforward']

