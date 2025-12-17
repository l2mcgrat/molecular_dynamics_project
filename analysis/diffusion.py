
import numpy as np

def mean_square_displacement(trajectory):
    """Compute MSD from trajectory positions."""
    Nsteps, N, _ = trajectory.shape
    msd = np.zeros(Nsteps)
    for t in range(Nsteps):
        disp = trajectory[t] - trajectory[0]
        msd[t] = np.mean(np.sum(disp**2, axis=1))
    return msd