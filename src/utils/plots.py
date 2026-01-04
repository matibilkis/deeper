"""
Plotting utilities for visualizing training progress and results.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
try:
    from src.quantum.receivers import ps_maxlik
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from src.quantum.receivers import ps_maxlik


def plot_learning_curves(rt, pt, optimal, directory, losses=None):
    """
    Plot learning curves showing reward and success probability over time.
    
    Args:
        rt: Cumulative average rewards
        pt: Success probabilities
        optimal: Optimal performance value
        directory: Directory to save plots
        losses: Optional list of loss values to plot
    """
    matplotlib.rc('font', serif='cm10')
    plt.rcParams.update({'font.size': 40})

    plt.figure(figsize=(20, 10), dpi=100)
    T = len(rt)
    
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.plot(np.log10(np.arange(1, T+1)), rt, color="red", linewidth=15, alpha=0.8, label=r'$R_t$')
    ax1.plot(np.log10(np.arange(1, T+1)), optimal * np.ones(T), color="black", linewidth=15, alpha=0.5, label="optimal")
    ax1.set_xlabel("Episode (log10)")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend()

    ax2.plot(np.log10(np.arange(1, T+1)), pt, color="blue", linewidth=15, alpha=0.8, label=r'$P_t$')
    ax2.plot(np.log10(np.arange(1, T+1)), optimal * np.ones(T), color="black", linewidth=15, alpha=0.5, label="optimal")
    ax2.set_xlabel("Episode (log10)")
    ax2.set_ylabel("Success Probability")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{directory}/learning_curves.png")
    plt.close()

    if losses is not None:
        plt.figure(figsize=(15, 10), dpi=100)
        for i, loss in enumerate(losses):
            plt.plot(np.arange(1, len(loss)+1), loss, '--', alpha=0.85, linewidth=5, label=f"Loss {i+1}")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{directory}/losses.png")
        plt.close()


def plot_beta_histogram(history_betas, history_betas_would_have_done, betas, directory):
    """
    Plot histogram of beta values used during training.
    
    Args:
        history_betas: List of beta values actually used
        history_betas_would_have_done: List of beta values that would have been used (greedy)
        betas: Array of all possible beta values
        directory: Directory to save plot
    """
    matplotlib.rc('font', serif='cm10')
    plt.rcParams.update({'font.size': 40})

    optimal_beta = betas[np.where(ps_maxlik(betas) == max(ps_maxlik(betas)))[0][0]]
    
    plt.figure(figsize=(20, 10), dpi=100)
    plt.hist(history_betas, bins=100, facecolor='r', alpha=0.6, edgecolor='blue', label="done")
    plt.hist(history_betas_would_have_done, bins=100, facecolor='g', alpha=0.4, edgecolor='black', label="would have done")
    plt.axvline(optimal_beta, color='black', linestyle='--', linewidth=3, label="optimal")
    plt.axvline(-optimal_beta, color='black', linestyle='--', linewidth=3)
    plt.xlabel(r'$\beta$')
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{directory}/histogram_betas.png")
    plt.close()

