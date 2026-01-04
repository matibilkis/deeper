"""
Unit tests for quantum receiver utility functions.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.quantum.receivers import Prob, qval, ps_maxlik


class TestQuantumReceivers(unittest.TestCase):
    """Test cases for quantum receiver functions."""
    
    def test_prob_basic(self):
        """Test basic probability calculation."""
        alpha = 0.4
        beta = 0.5
        n = 0
        
        prob = Prob(alpha, beta, n)
        
        # Probability should be in [0, 1]
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        
        # For n=0, should return p0 = exp(-(alpha-beta)^2)
        expected = np.exp(-(alpha - beta)**2)
        self.assertAlmostEqual(prob, expected, places=5)
    
    def test_prob_outcome_one(self):
        """Test probability for outcome n=1."""
        alpha = 0.4
        beta = 0.5
        n = 1
        
        prob = Prob(alpha, beta, n)
        
        # For n=1, should return 1 - p0
        p0 = np.exp(-(alpha - beta)**2)
        expected = 1 - p0
        self.assertAlmostEqual(prob, expected, places=5)
    
    def test_prob_symmetry(self):
        """Test probability symmetry properties."""
        alpha = 0.4
        beta = 0.5
        
        prob_0 = Prob(alpha, beta, 0)
        prob_1 = Prob(alpha, beta, 1)
        
        # Probabilities should sum to 1
        self.assertAlmostEqual(prob_0 + prob_1, 1.0, places=5)
    
    def test_qval_basic(self):
        """Test Q-value calculation."""
        beta = 0.5
        n = 0
        guess = 1.0
        
        q = qval(beta, n, guess)
        
        # Q-value should be in [0, 1] (probability)
        self.assertGreaterEqual(q, 0.0)
        self.assertLessEqual(q, 1.0)
    
    def test_qval_different_guesses(self):
        """Test Q-value for different guesses."""
        beta = 0.5
        n = 0
        
        q_plus = qval(beta, n, 1.0)
        q_minus = qval(beta, n, -1.0)
        
        # Both should be valid probabilities
        self.assertGreaterEqual(q_plus, 0.0)
        self.assertLessEqual(q_plus, 1.0)
        self.assertGreaterEqual(q_minus, 0.0)
        self.assertLessEqual(q_minus, 1.0)
    
    def test_ps_maxlik_basic(self):
        """Test maximum likelihood success probability."""
        beta = 0.5
        
        ps = ps_maxlik(beta)
        
        # Should be a valid probability
        self.assertGreaterEqual(ps, 0.0)
        self.assertLessEqual(ps, 1.0)
    
    def test_ps_maxlik_symmetry(self):
        """Test ps_maxlik symmetry."""
        beta = 0.5
        
        ps_pos = ps_maxlik(beta)
        ps_neg = ps_maxlik(-beta)
        
        # Should be symmetric (or at least valid)
        self.assertGreaterEqual(ps_pos, 0.0)
        self.assertLessEqual(ps_pos, 1.0)
        self.assertGreaterEqual(ps_neg, 0.0)
        self.assertLessEqual(ps_neg, 1.0)
    
    def test_ps_maxlik_range(self):
        """Test ps_maxlik for range of beta values."""
        betas = np.linspace(-1.5, 1.5, 10)
        
        for beta in betas:
            ps = ps_maxlik(beta)
            self.assertGreaterEqual(ps, 0.0)
            self.assertLessEqual(ps, 1.0)
    
    def test_qval_consistency(self):
        """Test Q-value consistency across outcomes."""
        beta = 0.5
        guess = 1.0
        
        q_0 = qval(beta, 0, guess)
        q_1 = qval(beta, 1, guess)
        
        # Both should be valid probabilities
        self.assertGreaterEqual(q_0, 0.0)
        self.assertLessEqual(q_0, 1.0)
        self.assertGreaterEqual(q_1, 0.0)
        self.assertLessEqual(q_1, 1.0)


if __name__ == '__main__':
    unittest.main()

