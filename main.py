
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
# For visualization
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------- Simulation Parameters ----------------
# User input: density in kg/m^3
potential_type = "coulomb"  # options: "coulomb", "dipole_lj"
input_density = 1000  # lower density to reduce overlaps
box_length = 1e-9  # increase box size to reduce overlaps
volume = box_length ** 3  # m^3
mass_water = 18 * 1.66054e-27  # kg per molecule
N = int(input_density * volume / mass_water)  # number of molecules
print(f"Simulating {N} molecules for density {input_density} kg/m^3 in box {box_length} m")
steps = 1000     # number of steps
ts = 1e-15      # timestep (s)

# Temperature sweep
T_list = np.arange(10, 411, 40)  # e.g., 10K to 410K in 40K steps

# ---------------- Simulation Functions ----------------

def run_simulation(system_class, N, T, ts, steps=500):
    system = system_class(N=N, T=T, ts=ts)
    trajectory = []
    for step in range(steps):
        system.step()
        trajectory.append(system.positions.copy())
    return np.array(trajectory), system

def save_trajectory(traj, N, steps, T):
    os.makedirs("trajectories", exist_ok=True)
    today = datetime.today().strftime("%Y-%m-%d")
    filename = f"trajectories/traj_{today}_N{N}_steps{steps}_T{T}K.npy"
    np.save(filename, traj)
    print("Saved:", filename)
    return filename

def extract_theta_phi(dipoles):
    """Return theta (polar) and phi (azimuthal) angles for dipole vectors."""
    norms = np.linalg.norm(dipoles, axis=1)
    # Avoid division by zero
    norms[norms == 0] = 1e-12
    theta = np.arccos(dipoles[:,2] / norms)
    phi = np.arctan2(dipoles[:,1], dipoles[:,0])
    return theta, phi

# ---------------- Main Temperature Sweep ----------------
# Select potential and set plot subfolders
if potential_type == "coulomb":
    from potentials.Molecular_Coulomb_Potential import CoulombWaterPotential
    SystemClass = CoulombWaterPotential
    plot_subfolder = "plots/Molecular_Coulomb_Potential"
elif potential_type == "dipole_lj":
    from potentials.DD_and_LJ_waters_in_a_box import DipoleDipole
    SystemClass = DipoleDipole
    plot_subfolder = "plots/Dipole-Dipole_and_LJ_Potential"
else:
    raise ValueError("Unknown potential_type")
theta_subfolder = os.path.join(plot_subfolder, "theta_density_plots")
phi_subfolder = os.path.join(plot_subfolder, "phi_density_plots")
pe_subfolder = os.path.join(plot_subfolder, "potential_energy_plots")
os.makedirs(theta_subfolder, exist_ok=True)
os.makedirs(phi_subfolder, exist_ok=True)
os.makedirs(pe_subfolder, exist_ok=True)

# Calculate actual density (kg/m^3) for reporting and PDF filenames
density = (mass_water * N) / volume  # kg/m^3
T_min = int(np.min(T_list))
T_max = int(np.max(T_list))
box_nm = box_length * 1e9  # box size in nm
theta_pdf_filename = f"{theta_subfolder}/theta_densityplots_{N}mol_{T_min}Kto{T_max}K_density{density:.1f}kgm3_box{box_nm:.2f}nm.pdf"
phi_pdf_filename = f"{phi_subfolder}/phi_densityplots_{N}mol_{T_min}Kto{T_max}K_density{density:.1f}kgm3_box{box_nm:.2f}nm.pdf"
pe_pdf_filename = f"{pe_subfolder}/potential_energy_{N}mol_{T_min}Kto{T_max}K_density{density:.1f}kgm3_box{box_nm:.2f}nm.pdf"

# Accumulate all density plots in a single PDF per type
potential_energies = []
theta_figs = []
phi_figs = []
pe_figs = []
for T in T_list:
    print(f"Running simulation at T={T}K...")
    traj, system = run_simulation(SystemClass, N, T, ts=1e-16, steps=steps)  # smaller timestep for stability
    filename = save_trajectory(traj, N, steps, T)
    # Track angles if possible, else skip
    if hasattr(system, 'dipoles'):
        dipoles = system.dipoles
        theta, phi = extract_theta_phi(dipoles)
    elif hasattr(system, 'get_atom_positions'):
        # For CoulombWaterPotential, track O-H bond angles for each molecule
        theta = np.zeros(N)
        phi = np.zeros(N)
        for i in range(N):
            atom_pos = system.get_atom_positions(system.positions[i], system.orientations[i])
            # O-H1 vector
            oh1 = atom_pos[1] - atom_pos[0]
            norm_oh1 = np.linalg.norm(oh1)
            if norm_oh1 == 0:
                continue
            theta[i] = np.arccos(oh1[2] / norm_oh1)
            phi[i] = np.arctan2(oh1[1], oh1[0])
    else:
        theta = phi = np.zeros(N)

    # Compute potential energy at each timestep
    pe_t = []
    for positions in traj:
        pe = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if hasattr(system, 'dipoles'):
                    r = positions[i] - positions[j]
                    pe += system.potential(r, system.dipoles[i], system.dipoles[j])
                elif hasattr(system, 'orientations'):
                    pe += system.potential(positions[i], positions[j], system.orientations[i], system.orientations[j])
                else:
                    pe += system.potential(positions[i], positions[j])
        pe_t.append(pe)

    # Percentile-based filter: remove top/bottom 1% of values (no hard cutoff)
    pe_t = np.array(pe_t)
    if len(pe_t) > 0:
        lower, upper = np.percentile(pe_t, [1, 99])
        mask2 = (pe_t >= lower) & (pe_t <= upper)
        pe_t = pe_t[mask2]
    # Convert to electron volts (1 eV = 1.60218e-19 J)
    pe_t_ev = pe_t / 1.60218e-19
    potential_energies.append((T, pe_t_ev))

    # Per-temperature potential energy plot
    fig_pe = plt.figure(figsize=(8,5))
    plt.plot(pe_t_ev)
    plt.xlabel('Timestep')
    plt.ylabel('Potential Energy (eV)')
    plt.title(f'Potential Energy at T={T}K\nBox={box_nm:.2f}nm, Density={density:.1f}kg/m^3')
    plt.tight_layout()
    pe_figs.append(fig_pe)

    # Theta density plot with variance check
    fig_theta = plt.figure(figsize=(7,5))
    if np.std(theta) > 1e-6 and len(np.unique(theta)) > 1:
        theta_grid = np.linspace(0, np.pi, 200)
        kde_theta = gaussian_kde(theta)
        plt.plot(theta_grid, kde_theta(theta_grid), color='navy')
        plt.xlabel('Theta (rad)')
        plt.ylabel('Density')
        plt.title(f'Theta density at T={T}K')
    else:
        plt.hist(theta, bins=20, color='navy', alpha=0.7)
        plt.xlabel('Theta (rad)')
        plt.ylabel('Count')
        plt.title(f'Theta (no variance) at T={T}K')
        print(f"[Warning] Skipping KDE for theta at T={T}K due to insufficient variance.")
    plt.tight_layout()
    theta_figs.append(fig_theta)

    # Phi density plot with variance check
    fig_phi = plt.figure(figsize=(7,5))
    if np.std(phi) > 1e-6 and len(np.unique(phi)) > 1:
        phi_grid = np.linspace(-np.pi, np.pi, 200)
        kde_phi = gaussian_kde(phi)
        plt.plot(phi_grid, kde_phi(phi_grid), color='darkred')
        plt.xlabel('Phi (rad)')
        plt.ylabel('Density')
        plt.title(f'Phi density at T={T}K')
    else:
        plt.hist(phi, bins=20, color='darkred', alpha=0.7)
        plt.xlabel('Phi (rad)')
        plt.ylabel('Count')
        plt.title(f'Phi (no variance) at T={T}K')
        print(f"[Warning] Skipping KDE for phi at T={T}K due to insufficient variance.")
    plt.tight_layout()
    phi_figs.append(fig_phi)

    print(f"Added density plots for T={T}K to PDFs.")

# Save all theta density plots to one PDF
with PdfPages(theta_pdf_filename) as theta_pdf:
    for fig in theta_figs:
        theta_pdf.savefig(fig)
        plt.close(fig)
# Save all phi density plots to one PDF
with PdfPages(phi_pdf_filename) as phi_pdf:
    for fig in phi_figs:
        phi_pdf.savefig(fig)
        plt.close(fig)

# Save all per-temperature potential energy plots to one PDF
with PdfPages(pe_pdf_filename) as pe_pdf:
    for fig in pe_figs:
        pe_pdf.savefig(fig)
        plt.close(fig)
print(f"Saved all per-temperature potential energy plots to {pe_pdf_filename}")

# ---------------- Main Execution ----------------

# Remove or guard the __main__ block to avoid duplicate runs and ValueError
# If you want to keep single-run functionality, use a flag or comment out the block below:
# if __name__ == "__main__":
#     traj_dd, _ = run_simulation(DipoleDipole, N=N, T=T, ts=ts, steps=steps)
#     filename = save_trajectory(traj_dd, N=N, steps=steps, T=T)
#
#     print("Trajectory shape:", traj_dd.shape)
#
#     # Example analysis
#     msd = mean_square_displacement(traj_dd)
#     print("MSD first 10 steps:", msd[:10])
#
#     r, g_r = radial_distribution(traj_dd, box_length=1e-9)
#     print("RDF sample:", list(zip(r[:10], g_r[:10])))

