
# Copilot Instructions for molecular_dynamics_project

## Project Overview
This project simulates molecular dynamics for dipolar water molecules in a 1nm box. The workflow is:

1. **Run simulation** via [main.py](../../main.py) (sets parameters, runs system, saves trajectory to [trajectories/](../../trajectories/)).
2. **Analyze results** using modular tools in [analysis/](../../analysis/) (diffusion, structure, energies, visualization).

## Architecture & Key Files
- **main.py**: Entry point. Sets simulation parameters (N, T, steps, ts), runs simulation (using a class from [potentials/](../../potentials/)), saves trajectory, and can call analysis functions.
- **potentials/**: Contains system classes (e.g., `DipoleDipole` in `DD_and_LJ_waters_in_a_box.py`) implementing state, integration, and force calculations. Required attributes: `.positions`, `.velocities`, `.dipoles`, `.potential()`.
- **analysis/**: Modular analysis tools:
  - [diffusion.py](../../analysis/diffusion.py): `mean_square_displacement(trajectory)`
  - [radial_distribution_function.py](../../analysis/radial_distribution_function.py): `radial_distribution(trajectory, box_length, bins)`
  - [energies.py](../../analysis/energies.py): `kinetic_energy(velocities, mass)`, `potential_energy(system, positions)`
  - [io.py](../../analysis/io.py): `load_trajectory(filename)` (loads `.npy` files)
  - [visualize_md_trajectory.py](../../analysis/visualize_md_trajectory.py): Visualization utilities
- **trajectories/**: Stores simulation output as `.npy` files, named with date, N, steps, and T (e.g., `traj_2026-01-17_N26_steps500_T250K.npy`).

## Patterns & Conventions
- All analysis functions expect NumPy arrays (trajectories, positions, velocities).
- Simulation parameters are set in [main.py](../../main.py) and passed explicitly to system/analysis functions.
- Trajectories are always saved/loaded as `.npy` (NumPy binary format).
- No external dependencies beyond NumPy (no matplotlib, pandas, etc.).
- All code assumes SI units (meters, seconds, Kelvin, Joules).

## Developer Workflows
**Run a simulation:**
1. Edit parameters in [main.py](../../main.py)
2. Run [main.py](../../main.py) to generate a trajectory file in [trajectories/](../../trajectories/)

**Analyze a trajectory:**
1. Use `from analysis.io import load_trajectory` to load a `.npy` file
2. Call analysis functions (e.g., `mean_square_displacement`, `radial_distribution`, `kinetic_energy`, `potential_energy`) with loaded data

**Add new analysis:**
- Place new scripts in [analysis/](../../analysis/), follow function signature conventions (input: NumPy arrays)

## Examples
- Compute MSD: `from analysis.diffusion import mean_square_displacement`
- Compute RDF: `from analysis.radial_distribution_function import radial_distribution`
- Compute energies: `from analysis.energies import kinetic_energy, potential_energy`

## Special Notes
- No explicit test suite; validate by comparing analysis outputs to expected physical behavior
- No build step required; run directly with Python 3 and NumPy installed

## Project-Specific Tips
- Visualization scripts are in [analysis/visualize_md_trajectory.py](../../analysis/visualize_md_trajectory.py) and [analysis/dynamics_videos/](../../analysis/dynamics_videos/)
- Plot outputs are organized in [plots/](../../plots/) by potential and property
- System classes in [potentials/](../../potentials/) may be extended for new interaction models; follow attribute conventions
