"""
Miscellaneous utility functions.
"""

import os


def record():
    """
    Record and increment run number for experiment tracking.
    
    Creates or updates a file to track experiment run numbers.
    
    Returns:
        Current run number
    """
    if not os.path.exists("results/number_rune.txt"):
        with open("results/number_rune.txt", "w+") as f:
            f.write("0")
            f.close()
        number_run = 0
    else:
        with open("results/number_rune.txt", "r") as f:
            a = f.readlines()[0]
            f.close()
        with open("results/number_rune.txt", "w") as f:
            f.truncate(0)
            f.write(str(int(a) + 1))
            f.close()
        number_run = int(a) + 1
    return number_run

