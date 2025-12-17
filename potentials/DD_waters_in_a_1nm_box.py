
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
        # simple dipoles aligned along z-axis
        self.dipoles = np.tile(np.array([0,0,1e-30]), (N,1))

    def init_positions(self):
        return np.random.rand(self.N, 3) * self.L

    def init_velocities(self):
        sigma = np.sqrt(self.kB * self.T / self.mass)
        return np.random.normal(0, sigma, (self.N, 3))

    def potential(self, r, mu_i, mu_j):
        r_mag = np.linalg.norm(r)
        if r_mag == 0: return 0.0
        r_hat = r / r_mag
        mu_dot = np.dot(mu_i, mu_j)
        mu_r_i = np.dot(mu_i, r_hat)
        mu_r_j = np.dot(mu_j, r_hat)
        prefactor = 1/(4*np.pi*self.epsilon0) * 1/(r_mag**3)
        return prefactor * (mu_dot - 3*mu_r_i*mu_r_j)

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

