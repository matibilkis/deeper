"""
Unit tests for miscellaneous utilities.
"""

import unittest
import os
import sys
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.misc import record


class TestMisc(unittest.TestCase):
    """Test cases for miscellaneous utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary results directory for testing
        self.test_results_dir = "test_results"
        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        # Change to test directory context
        self.original_dir = os.getcwd()
        os.chdir(self.test_results_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        os.chdir(self.original_dir)
        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
    
    def test_record_initial(self):
        """Test record function creates file and returns 0 initially."""
        if os.path.exists("number_rune.txt"):
            os.remove("number_rune.txt")
        
        run_number = record()
        
        self.assertEqual(run_number, 0)
        self.assertTrue(os.path.exists("number_rune.txt"))
        
        # Check file content
        with open("number_rune.txt", "r") as f:
            content = f.read().strip()
            self.assertEqual(content, "0")
    
    def test_record_increments(self):
        """Test record function increments run number."""
        if os.path.exists("number_rune.txt"):
            os.remove("number_rune.txt")
        
        # First call
        run_1 = record()
        self.assertEqual(run_1, 0)
        
        # Second call
        run_2 = record()
        self.assertEqual(run_2, 1)
        
        # Third call
        run_3 = record()
        self.assertEqual(run_3, 2)
        
        # Check file content
        with open("number_rune.txt", "r") as f:
            content = f.read().strip()
            self.assertEqual(content, "2")


if __name__ == '__main__':
    unittest.main()

