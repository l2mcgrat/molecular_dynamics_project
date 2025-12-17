
import numpy as np

def kinetic_energy(velocities, mass):
    """Compute average kinetic energy per particle."""
    KE = 0.5 * mass * np.sum(velocities**2, axis=1)
    return np.mean(KE)

def potential_energy(system, positions):
    """Compute total potential energy using system's potential method."""
    PE = 0.0
    for i in range(system.N):
        for j in range(i+1, system.N):
            r = positions[i] - positions[j]
            PE += system.potential(r, system.dipoles[i], system.dipoles[j])
    return PE