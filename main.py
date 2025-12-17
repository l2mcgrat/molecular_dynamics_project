from potentials.DD_waters_in_a_1nm_box import DipoleDipole
import numpy as np
import os
from datetime import datetime

# ---------------- Simulation Parameters ----------------
N = 50          # number of molecules
T = 300         # temperature (K)
steps = 500     # number of steps
ts = 1e-15      # timestep (s)

# ---------------- Simulation Functions ----------------
def run_simulation(system_class, N, T, ts, steps=1000):
    system = system_class(N=N, T=T, ts=ts)
    trajectory = []
    for step in range(steps):
        system.step()
        trajectory.append(system.positions.copy())
    return np.array(trajectory)

def save_trajectory(traj, N, steps, T):
    os.makedirs("trajectories", exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")
    filename = f"trajectories/traj_{today}_N{N}_steps{steps}_T{T}K.npy"
    np.save(filename, traj)
    print("Saved:", filename)
    return filename

# ---------------- Analysis Functions ----------------
def mean_square_displacement(trajectory):
    """Compute MSD from trajectory positions."""
    Nsteps, N, _ = trajectory.shape
    msd = np.zeros(Nsteps)
    for t in range(Nsteps):
        disp = trajectory[t] - trajectory[0]
        msd[t] = np.mean(np.sum(disp**2, axis=1))
    return msd

def radial_distribution(trajectory, box_length, bins=100):
    """Compute RDF from trajectory positions."""
    N = trajectory.shape[1]
    dr = box_length / bins
    rdf = np.zeros(bins)
    for frame in trajectory:
        for i in range(N):
            for j in range(i+1, N):
                r = np.linalg.norm(frame[i] - frame[j])
                bin_index = int(r/dr)
                if bin_index < bins:
                    rdf[bin_index] += 2
    # Normalize
    norm = N * (N-1) / 2 * len(trajectory)
    rdf /= norm
    r_values = np.linspace(0, box_length, bins)
    return r_values, rdf

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    traj_dd = run_simulation(DipoleDipole, N=N, T=T, ts=ts, steps=steps)
    filename = save_trajectory(traj_dd, N=N, steps=steps, T=T)

    print("Trajectory shape:", traj_dd.shape)

    # Example analysis
    msd = mean_square_displacement(traj_dd)
    print("MSD first 10 steps:", msd[:10])

    r, g_r = radial_distribution(traj_dd, box_length=1e-9)
    print("RDF sample:", list(zip(r[:10], g_r[:10])))

