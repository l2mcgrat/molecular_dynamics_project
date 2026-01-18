
import numpy as np

class DipoleDipole:
    
    kB = 1.380649e-23       # J/K
    epsilon0 = 8.854e-12    # F/m

    def __init__(self, N, T, ts, box_length=1e-9, mass=18 * 1.66054e-27):
        self.N = N
        self.T = T
        self.ts = ts
        self.L = box_length
        self.mass = mass
        self.positions = self.init_positions()
        self.velocities = self.init_velocities()
        # initialize all dipoles aligned along z-axis, real water dipole moment
        self.dipole_moment = 6.2e-30  # C·m (real water)
        self.dipoles = np.tile(np.array([0,0,1]), (self.N,1)) * self.dipole_moment
        # initialize angular velocities for each dipole (random, thermalized)
        self.ang_velocities = self.init_angular_velocities()

    def init_angular_velocities(self):
        # Assume moment of inertia I for water molecule (approximate)
        I = 2.99e-47  # kg m^2 (about right for H2O)
        sigma = np.sqrt(self.kB * self.T / I)
        return np.random.normal(0, sigma, (self.N, 3))

    # No longer used: all dipoles start aligned

    def init_positions(self):
        return np.random.rand(self.N, 3) * self.L

    def init_velocities(self):
        sigma = np.sqrt(self.kB * self.T / self.mass)
        return np.random.normal(0, sigma, (self.N, 3))

    def potential(self, r, mu_i, mu_j):
        r_mag = np.linalg.norm(r)
        if r_mag == 0: return 0.0
        # Lennard-Jones repulsive core (parameters for O-O in water, but only repulsive part)
        sigma = 3.2e-10  # m (O-O distance)
        epsilon = 0.65e-21  # J (depth, but we only want repulsion)
        lj_rep = 4 * epsilon * ( (sigma/r_mag)**12 ) if r_mag < 2.5*sigma else 0.0
        r_hat = r / r_mag
        mu_dot = np.dot(mu_i, mu_j)
        mu_r_i = np.dot(mu_i, r_hat)
        mu_r_j = np.dot(mu_j, r_hat)
        prefactor = 1/(4*np.pi*self.epsilon0) * 1/(r_mag**3)
        dipole_term = prefactor * (mu_dot - 3*mu_r_i*mu_r_j)
        return dipole_term + lj_rep

    def forces(self):
        F = np.zeros_like(self.positions)
        for i in range(self.N):
            for j in range(i+1, self.N):
                r = self.positions[i] - self.positions[j]
                r_mag = np.linalg.norm(r)
                if r_mag == 0: continue
                # gradient of potential wrt r gives force
                mu_i, mu_j = self.dipoles[i], self.dipoles[j]
                prefactor = 1/(4*np.pi*self.epsilon0) * 1/(r_mag**4)
                r_hat = r / r_mag
                mu_dot = np.dot(mu_i, mu_j)
                mu_r_i = np.dot(mu_i, r_hat)
                mu_r_j = np.dot(mu_j, r_hat)
                f_vec = prefactor * (3*mu_r_i*mu_r_j - mu_dot) * r_hat
                F[i] += f_vec
                F[j] -= f_vec
        return F

    def apply_boundary(self):
        self.positions %= self.L

    def step(self):
        # Velocity-Verlet integration 
        F = self.forces()
        self.positions += self.velocities * self.ts + 0.5 * F/self.mass * self.ts**2
        F_new = self.forces()
        self.velocities += 0.5 * (F + F_new)/self.mass * self.ts
        self.apply_boundary()
        # Langevin thermostat (simple): add friction and random force to velocities
        gamma = 1e13  # friction coefficient (1/s)
        sigma_v = np.sqrt(2 * gamma * self.kB * self.T / self.mass / self.ts)
        self.velocities *= np.exp(-gamma * self.ts)
        self.velocities += np.random.normal(0, sigma_v, self.velocities.shape)
        # Rotational molecular dynamics: update dipole orientations by integrating angular velocity
        self.update_dipole_orientations()

    def update_dipole_orientations(self):
        # For each dipole, rotate by ang_velocity * ts (Rodrigues' formula)
        for i in range(self.N):
            omega = self.ang_velocities[i]
            angle = np.linalg.norm(omega) * self.ts
            if angle == 0:
                continue
            axis = omega / np.linalg.norm(omega)
            v = self.dipoles[i] / np.linalg.norm(self.dipoles[i])
            v_rot = (v * np.cos(angle) +
                     np.cross(axis, v) * np.sin(angle) +
                     axis * np.dot(axis, v) * (1 - np.cos(angle)))
            self.dipoles[i] = v_rot * 1e-30

    def randomize_dipoles_by_temperature(self):
        # The higher the temperature, the more randomization (simple model)
        # For each dipole, apply a small random rotation
        # Make angle_scale much smaller at low T, much larger at high T
        # e.g., angle_scale = max_angle * (T / (T + T0)), with T0 ~ 50K
        max_angle = np.pi  # up to 180 degrees at very high T
        T0 = 50.0  # controls sharpness of transition
        angle_scale = max_angle * (self.T / (self.T + T0))
        for i in range(self.N):
            # Generate a random axis
            axis = np.random.normal(0, 1, 3)
            axis /= np.linalg.norm(axis)
            # Generate a small random angle
            angle = np.random.normal(0, angle_scale)
            # Rodrigues' rotation formula
            v = self.dipoles[i] / np.linalg.norm(self.dipoles[i])
            v_rot = (v * np.cos(angle) +
                     np.cross(axis, v) * np.sin(angle) +
                     axis * np.dot(axis, v) * (1 - np.cos(angle)))
            self.dipoles[i] = v_rot * 1e-30

