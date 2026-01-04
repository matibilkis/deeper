# Testing Guide

This document describes the testing setup and how to run tests for the quantum-receiver-rl package.

## Test Results Summary

✅ **Quantum Receiver Functions**: All 9 tests passing  
✅ **Import Tests**: Core imports working (TensorFlow/Matplotlib optional)  
✅ **Code Structure**: All modules properly organized and importable  

## Running Tests

### Quick Test (No Dependencies Required)

Test quantum functions (only requires NumPy):
```bash
python tests/test_quantum/test_receivers.py -v
```

### Full Test Suite (Requires Dependencies)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run all tests**:
   ```bash
   # Using unittest (built-in)
   python tests/run_tests.py
   
   # Or using unittest directly
   python -m unittest discover tests -v
   
   # Using pytest (if installed)
   pytest tests/ -v
   ```

3. **Run specific test categories**:
   ```bash
   # Test networks
   python -m unittest tests.test_networks -v
   
   # Test algorithms
   python -m unittest tests.test_algorithms -v
   
   # Test quantum functions
   python -m unittest tests.test_quantum -v
   
   # Test utilities
   python -m unittest tests.test_utils -v
   
   # Integration tests
   python -m unittest tests.test_integration -v
   ```

## Test Coverage

### ✅ Quantum Receiver Functions (`test_quantum/test_receivers.py`)
- `Prob()`: Probability calculations
- `qval()`: Q-value computations  
- `ps_maxlik()`: Maximum likelihood probabilities
- **Status**: All 9 tests passing ✅

### ✅ Networks (`test_networks/`)
- `test_actor.py`: Actor network initialization, forward pass, target updates
- `test_critic.py`: Critic network (LSTM and feedforward), sequence processing
- **Status**: Tests created, require TensorFlow to run

### ✅ Algorithms (`test_algorithms/`)
- `test_buffer.py`: Replay buffer operations, sampling, overflow handling
- **Status**: Tests created, require TensorFlow to run

### ✅ Utilities (`test_utils/`)
- `test_misc.py`: Utility functions like `record()`
- **Status**: Tests created

### ✅ Integration (`test_integration.py`)
- Tests components working together
- **Status**: Tests created, require TensorFlow to run

### ✅ Imports (`test_imports.py`)
- Verifies all modules can be imported
- Gracefully handles missing dependencies
- **Status**: Working ✅

## Test Structure

```
tests/
├── __init__.py
├── run_tests.py              # Test runner script
├── test_imports.py           # Import verification tests
├── test_integration.py       # Integration tests
├── test_networks/            # Network tests
│   ├── test_actor.py
│   └── test_critic.py
├── test_algorithms/          # Algorithm tests
│   └── test_buffer.py
├── test_quantum/             # Quantum function tests
│   └── test_receivers.py
└── test_utils/               # Utility tests
    └── test_misc.py
```

## Continuous Integration

A GitHub Actions workflow is configured in `.github/workflows/tests.yml` that will:
- Run tests on Python 3.7, 3.8, and 3.9
- Install dependencies
- Run all tests automatically on push/PR

## Test Requirements

### Minimal (for quantum tests only)
- Python 3.7+
- NumPy

### Full (for all tests)
- Python 3.7+
- TensorFlow >= 2.0.0
- NumPy >= 1.18.0
- Matplotlib >= 3.0.0
- pytest >= 6.0.0 (optional, for pytest runner)

## Known Issues

- Some tests require TensorFlow to be installed
- Tests will skip TensorFlow-dependent tests if TensorFlow is not available
- Matplotlib is optional for import tests but required for plotting functions

## Adding New Tests

When adding new functionality:

1. Create test file in appropriate subdirectory: `tests/test_<module>/test_<name>.py`
2. Follow existing test patterns
3. Use descriptive test names
4. Add docstrings to test classes and methods
5. Run tests to ensure they pass

Example:
```python
import unittest
from src.your_module import YourClass

class TestYourClass(unittest.TestCase):
    def test_something(self):
        # Your test code
        self.assertEqual(expected, actual)
```

## Test Best Practices

- ✅ Test edge cases
- ✅ Test error handling
- ✅ Use descriptive test names
- ✅ Keep tests independent
- ✅ Clean up after tests (use `setUp`/`tearDown`)
- ✅ Test both success and failure cases

---

**Last Updated**: Tests created and quantum function tests verified working.

