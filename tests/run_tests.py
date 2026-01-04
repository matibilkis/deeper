"""
Test runner for all unit tests.

Can be run with: python tests/run_tests.py
"""

import unittest
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def discover_and_run_tests():
    """Discover and run all tests."""
    # Discover tests in the tests directory
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.dirname(os.path.abspath(__file__)),
        pattern='test_*.py',
        top_level_dir=project_root
    )
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = discover_and_run_tests()
    sys.exit(exit_code)

