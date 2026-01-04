"""
Example training script for DDPG on quantum receiver optimization.

This script demonstrates how to use the DDPG implementation to train
an agent for optimizing quantum receiver strategies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from src.algorithms.ddpg import ddpg_kennedy


if __name__ == "__main__":
    # Example training configuration
    print("Starting DDPG training for quantum receiver optimization...")
    
    # Run training with default parameters
    result_dir = ddpg_kennedy(
        special_name="example_run",
        total_episodes=2000,
        buffer_size=2*10**6,
        batch_size=64,
        ep_guess=0.1,
        noise_displacement=0.5,
        lr_actor=0.001,
        lr_critic=0.0001,
        tau=0.005,
        plots=True
    )
    
    print(f"\nTraining completed! Results saved in: {result_dir}")

