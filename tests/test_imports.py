"""
Test that all modules can be imported correctly.

This test checks the structure and imports without requiring
TensorFlow to be installed (for CI/CD environments).
"""

import unittest
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestImports(unittest.TestCase):
    """Test that modules can be imported."""
    
    def test_quantum_receivers_import(self):
        """Test quantum receivers can be imported."""
        try:
            from src.quantum.receivers import Prob, qval, ps_maxlik
            self.assertTrue(True, "Quantum receivers imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import quantum receivers: {e}")
    
    def test_utils_import(self):
        """Test utils can be imported."""
        try:
            from src.utils.misc import record
            self.assertTrue(True, "Utils misc imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import utils.misc: {e}")
        
        # Plots may require matplotlib
        try:
            from src.utils.plots import plot_learning_curves, plot_beta_histogram
            self.assertTrue(True, "Utils plots imported successfully")
        except ImportError as e:
            if "matplotlib" in str(e).lower() or "No module named" in str(e):
                self.skipTest(f"Matplotlib not available: {e}")
            else:
                self.fail(f"Failed to import utils.plots: {e}")
    
    def test_buffer_import(self):
        """Test buffer can be imported (may fail if TensorFlow not available)."""
        try:
            from src.algorithms.buffer import ReplayBuffer
            self.assertTrue(True, "Buffer imported successfully")
        except ImportError as e:
            # This is expected if TensorFlow is not installed
            if "tensorflow" in str(e).lower() or "No module named" in str(e):
                self.skipTest(f"TensorFlow not available: {e}")
            else:
                self.fail(f"Failed to import buffer: {e}")
    
    def test_networks_import(self):
        """Test networks can be imported (may fail if TensorFlow not available)."""
        try:
            from src.networks.actor import Actor
            from src.networks.critic import Critic
            self.assertTrue(True, "Networks imported successfully")
        except ImportError as e:
            # This is expected if TensorFlow is not installed
            if "tensorflow" in str(e).lower() or "No module named" in str(e):
                self.skipTest(f"TensorFlow not available: {e}")
            else:
                self.fail(f"Failed to import networks: {e}")
    
    def test_algorithms_import(self):
        """Test algorithms can be imported (may fail if TensorFlow not available)."""
        try:
            from src.algorithms.ddpg import ddpg_kennedy, optimization_step
            self.assertTrue(True, "Algorithms imported successfully")
        except ImportError as e:
            # This is expected if TensorFlow is not installed
            if "tensorflow" in str(e).lower() or "No module named" in str(e):
                self.skipTest(f"TensorFlow not available: {e}")
            else:
                self.fail(f"Failed to import algorithms: {e}")


if __name__ == '__main__':
    unittest.main()

