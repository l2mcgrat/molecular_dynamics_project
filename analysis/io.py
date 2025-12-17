
import numpy as np

def load_trajectory(filename):
    """Load trajectory from .npy file."""
    return np.load(filename)