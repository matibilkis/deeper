"""
Quantum receiver utility functions.

Implements probability calculations and Q-value computations
for quantum receiver optimization, specifically for Kennedy and Dolinar receivers.
"""

import numpy as np


def Prob(alpha, beta, n):
    """
    Calculate outcome probability given displacement parameters.
    
    The probability is given by the overlap |<0|alpha - beta>|^2
    where alpha is the coherent state amplitude and beta is the displacement.
    
    Args:
        alpha: Coherent state amplitude
        beta: Displacement parameter
        n: Outcome (0 or 1)
        
    Returns:
        Probability of outcome n given alpha and beta
    """
    p0 = np.exp(-(alpha - beta)**2)
    if n == 0:
        return p0
    else:
        return 1 - p0


def qval(beta, n, guess):
    """
    Calculate Q-value: p(R=1 | g, n; beta).
    
    This is the probability of success given guess g, outcome n, and displacement beta.
    Uses Dolinar guessing rule (max-likelihood for L=1).
    
    Args:
        beta: Displacement parameter
        n: Measurement outcome (0 or 1)
        guess: Agent's guess (-1 or 1)
        
    Returns:
        Q-value (success probability)
    """
    # Dolinar guessing rule (= max-likelihood for L=1, careful sign of beta)
    alpha = 0.4
    pn = np.sum([Prob(g * alpha, beta, n) for g in [-1, 1]])
    return Prob(guess * alpha, beta, n) / pn


def ps_maxlik(beta):
    """
    Calculate maximum likelihood success probability.
    
    Dolinar guessing rule for L=1 (careful with sign of beta).
    
    Args:
        beta: Displacement parameter
        
    Returns:
        Maximum likelihood success probability
    """
    # Dolinar guessing rule (= max-likelihood for L=1, careful sign of beta)
    alpha = 0.4
    p = 0
    for n1 in [0, 1]:
        p += Prob(np.sign(beta) * (-1)**(n1) * alpha, beta, n1)
    return p / 2

