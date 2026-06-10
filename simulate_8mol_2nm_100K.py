"""
Simulation: 8 water molecules in a 2 nm box at T = 100 K.

Physics:
  - Dipole-dipole interactions between all molecule pairs (plus LJ
    repulsive core between molecular centres).
  - Lennard-Jones 12-6 wall potential: each molecule experiences a full
    LJ interaction with each of the six box faces whenever its
    centre-of-mass is within 0.5 nm of that face.
  - Translational velocities initialised from a Maxwell-Boltzmann
    distribution at 100 K (component-wise Gaussian with
    sigma = sqrt(kB * T / m)).
  - Rotational (dipole) degrees of freedom thermalised at 100 K.
  - Velocity-Verlet integrator with a Langevin thermostat.
  - Reflective (elastic) boundary conditions — the box acts as a
    physical container backed by the wall LJ repulsion.

Output:
  Trajectory saved to trajectories/ as a .npy file.
"""

import numpy as np
import os
from datetime import datetime

from potentials.DipoleDipole_WallLJ import DipoleDipoleWallLJ

# ---- Simulation parameters -----------------------------------------------
N = 8
T = 100.0       # K
box_length = 2e-9   # 2 nm
ts = 1e-15      # s  (1 fs timestep)
steps = 1000

# ---- Initialise system ----------------------------------------------------
print(f"=== DipoleDipole + LJ Wall simulation ===")
print(f"  Molecules : {N}")
print(f"  Box size  : {box_length * 1e9:.1f} nm")
print(f"  Temperature: {T} K (Maxwell-Boltzmann velocity init)")
print(f"  Wall LJ cutoff: 0.5 nm")
print(f"  Timestep  : {ts:.0e} s")
print(f"  Steps     : {steps}")
print()

system = DipoleDipoleWallLJ(N=N, T=T, ts=ts, box_length=box_length)

# ---- Run simulation -------------------------------------------------------
trajectory = []
for step in range(steps):
    system.step()
    trajectory.append(system.positions.copy())
    if step % 100 == 0:
        print(f"  Step {step:4d}/{steps}")

print(f"  Step {steps}/{steps}  — done\n")

traj = np.array(trajectory)  # shape: (steps, N, 3)

# ---- Save trajectory ------------------------------------------------------
os.makedirs("trajectories", exist_ok=True)
today = datetime.today().strftime("%Y-%m-%d")
filename = (
    f"trajectories/traj_{today}_N{N}_steps{steps}_T{int(T)}K_wallLJ.npy"
)
np.save(filename, traj)
print(f"Trajectory saved: {filename}")
print(f"Shape: {traj.shape}  (steps × molecules × xyz)")
