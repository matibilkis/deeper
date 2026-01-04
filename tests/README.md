# Tests

This directory contains unit tests for the deeper package.

## Quick Start

Run quantum function tests (no dependencies required):
```bash
python tests/test_quantum/test_receivers.py -v
```

Run all tests (requires dependencies):
```bash
pip install -r requirements.txt
python tests/run_tests.py
```

## Running Tests

### Using unittest (built-in, no extra dependencies)

```bash
# Run all tests
python tests/run_tests.py

# Run specific test file
python tests/test_quantum/test_receivers.py

# Run with verbose output
python -m unittest discover tests -v
```

### Using pytest (requires pytest installation)

```bash
# Install pytest (if not already installed)
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_quantum/test_receivers.py -v
```

## Test Structure

- `test_networks/`: Tests for Actor and Critic neural networks
- `test_algorithms/`: Tests for DDPG algorithm and replay buffer
- `test_quantum/`: Tests for quantum receiver utility functions
- `test_utils/`: Tests for utility functions
- `test_integration.py`: Integration tests for components working together
- `test_imports.py`: Tests that verify all modules can be imported

## Test Coverage

The tests cover:

✅ **Networks**: Actor and Critic initialization, forward passes, target network updates  
✅ **Algorithms**: Buffer operations, sampling, experience storage  
✅ **Quantum**: Probability calculations, Q-values, maximum likelihood  
✅ **Utils**: Utility functions and plotting  
✅ **Integration**: Components working together  

## Notes

- Some tests require TensorFlow to be installed (network tests)
- Tests that don't require TensorFlow will run even if it's not installed
- Import tests will skip TensorFlow-dependent tests if TensorFlow is not available

