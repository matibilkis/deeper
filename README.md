# Quantum Receiver RL: Deep Reinforcement Learning for Quantum Receiver Optimization

This repository contains experimental implementations of Deep Deterministic Policy Gradient (DDPG) algorithms for optimizing quantum communication receivers, building upon methods developed in related research on quantum receiver design and reinforcement learning.

## Overview

This project explores the application of deep reinforcement learning, specifically DDPG, to optimize quantum receiver strategies. The work extends concepts from quantum receiver theory (Kennedy receiver, Dolinar receiver) and applies modern deep learning techniques using TensorFlow/Keras.

**Note**: This is experimental research code. The DDPG implementation is computationally demanding and some components may require further development or optimization.

## Related Work

This repository builds upon and aims to improve methods from:

- [arXiv:2404.10726](https://arxiv.org/abs/2404.10726) - Recent work on quantum receiver optimization
- [arXiv:2203.09807](https://arxiv.org/abs/2203.09807) - Quantum communication receiver methods
- [arXiv:2001.10283](https://arxiv.org/abs/2001.10283) - Foundational quantum receiver research
- [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) - DDPG: Continuous Control with Deep Reinforcement Learning

Related implementations:
- [qrec](https://github.com/matibilkis/qrec) - Quantum receiver implementations
- [marek](https://github.com/matibilkis/marek) - Related quantum communication work

## Repository Structure

The repository contains multiple experimental branches exploring different approaches:

- **recurrent**: RNN-based critic network implementations with LSTM layers
- **DDPG_KENNEDY**: DDPG implementation focused on Kennedy receiver optimization
- **conti**: Continuous action space experiments
- **reward_matters**: Reward function variations
- **bernoulli**: Bernoulli action space experiments
- **fullKen**: Full Kennedy receiver implementations
- **RDPG-DOLINAR**: Recurrent DDPG for Dolinar receiver
- **FullSupervisedRNN**: Supervised learning baselines with RNNs
- **heroes**: Additional experimental approaches

## Technical Implementation

### TensorFlow/Keras Architecture

The implementation uses TensorFlow 2.x with Keras for building neural networks:

- **Actor Network**: Policy network that outputs continuous actions
  - Multi-layer dense networks with ReLU activations
  - Custom weight initialization and regularization (L1/L2)
  - Tanh output layer for bounded action spaces

- **Critic Network**: Q-value estimation network
  - LSTM layers for sequence processing (recurrent variants)
  - Masking layers for variable-length sequences
  - Dense layers with regularization
  - Handles multi-layer receiver scenarios

- **Replay Buffer**: Experience replay for off-policy learning
  - Deque-based circular buffer
  - Batch sampling for training
  - Test dataset generation for evaluation

### Key Features

- **Eager Execution**: TensorFlow eager mode for dynamic computation
- **Gradient Computation**: Custom gradient calculations for actor-critic updates
- **Target Networks**: Soft target network updates (tau parameter)
- **TensorBoard Integration**: Logging for training monitoring
- **Regularization**: L1/L2 regularization on network weights and activities

### Code Organization

The main components include:

- `nets.py`: Neural network definitions (Actor, Critic classes)
- `buffer.py`: Experience replay buffer implementation
- `main.py`: Training loops and optimization steps
- `misc.py`: Utility functions (probability calculations, Q-value computations)
- `plots.py`: Visualization and plotting utilities

## Installation

```bash
# Clone the repository
git clone https://github.com/matibilkis/quantum-receiver-rl.git
cd quantum-receiver-rl

# Install dependencies (example - adjust versions as needed)
pip install tensorflow>=2.0.0
pip install numpy pandas matplotlib tqdm
```

## Usage

The code structure supports various experimental configurations. Main training scripts are typically found in `main.py` files across different branches.

Example usage (from recurrent branch):
```python
from main import ddpgKennedy

# Run training with custom parameters
ddpgKennedy(
    total_episodes=2000,
    buffer_size=2*10**6,
    batch_size=64,
    lr_critic=0.0001,
    lr_actor=0.001,
    tau=0.005,
    noise_displacement=0.5
)
```

## Current Status

This repository represents experimental work on applying DDPG to quantum receiver optimization. The implementation includes:

✅ **TensorFlow/Keras neural network architectures** - Actor and Critic networks with LSTM support  
✅ **Actor-Critic framework** - DDPG algorithm with target networks and soft updates  
✅ **Experience replay buffer** - Efficient off-policy learning with batch sampling  
✅ **Gradient computation** - Custom optimization loops with TensorFlow eager execution  
✅ **Recurrent architectures** - LSTM-based critics for sequence processing  
✅ **TensorBoard integration** - Training monitoring and logging  
✅ **Comprehensive test suite** - Unit tests for all major components  

⚠️ **Note**: The full DDPG implementation is computationally demanding and may require further optimization. Some experimental branches contain incomplete or exploratory code.

## Research Context

The work focuses on optimizing quantum receiver strategies where:
- The agent learns to make optimal guesses based on quantum measurement outcomes
- Actions correspond to receiver configurations (e.g., displacement amplitudes)
- Rewards are based on successful state discrimination
- The problem involves sequential decision-making with quantum noise

## Contributing

This is a research repository. For questions or collaboration, please open an issue or contact the maintainer.

## License

[Specify your license here]

## Author

Matias Bilkis - [GitHub](https://github.com/matibilkis)

---

*This repository documents experimental work on deep reinforcement learning for quantum communication systems.*

