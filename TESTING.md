# Testing Guide

This document provides a quick overview of testing. For detailed information, see [tests/README.md](tests/README.md).

**Note**: This is a condensed version. For complete testing documentation, see [tests/README.md](tests/README.md).

# Testing Guide

Quick reference for running tests. For detailed documentation, see [tests/README.md](tests/README.md).

## Quick Start

**Run all tests:**
```bash
pip install -r requirements.txt
python tests/run_tests.py
```

**Quick test (no dependencies):**
```bash
python tests/test_quantum/test_receivers.py -v
```

## Test Status

✅ **Quantum Receiver Functions**: All 9 tests passing  
✅ **Import Tests**: Core imports working (TensorFlow/Matplotlib optional)  
✅ **Code Structure**: All modules properly organized and importable  

## Test Coverage

- **Networks**: Actor and Critic initialization, forward passes, target updates
- **Algorithms**: Buffer operations, sampling, experience storage
- **Quantum**: Probability calculations, Q-values, maximum likelihood
- **Utils**: Utility functions and plotting
- **Integration**: Components working together

## CI/CD

Tests run automatically on push/PR via GitHub Actions (see `.github/workflows/tests.yml`).

For more details, see [tests/README.md](tests/README.md).

