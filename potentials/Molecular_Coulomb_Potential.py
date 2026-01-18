import numpy as np

class CoulombWaterPotential:
    """
    Each water molecule is modeled as 3 point charges:
    - Oxygen: -2*delta_q at the origin
    - Two Hydrogens: +delta_q each, at fixed geometry (0.9572 Å, 104.52°)
    """
    # Lennard-Jones parameters (approximate for O, H)
    sigma_O = 3.166e-10  # meters (Oxygen)
    sigma_H = 2.958e-10  # meters (Hydrogen)
    epsilon_O = 0.650e-21  # J (Oxygen)
    epsilon_H = 0.065e-21  # J (Hydrogen, much weaker)
    min_dist = 0.7e-10  # meters (hard-sphere minimum)
    k_e = 8.987551787e9  # N m^2 / C^2 (Coulomb's constant)
    e = 1.602176634e-19  # elementary charge (C)

    def __init__(self, N, T, ts, box_length=1e-9, mass=18 * 1.66054e-27, delta_q=0.8476*1.602176634e-19):
        self.N = N
        self.T = T
        self.ts = ts
        self.L = box_length
        self.mass = mass
        self.delta_q = delta_q  # partial charge in C
        self.positions = self.init_positions_grid()
        self.velocities = self.init_velocities()
        # Water geometry: O at (0,0,0), H at (x1, y1, 0), H at (x2, y2, 0)
        angle = np.deg2rad(104.52)
        r_OH = 0.9572e-10  # meters
        self.rel_H1_0 = np.array([r_OH, 0, 0])
        self.rel_H2_0 = np.array([
            r_OH * np.cos(angle),
            r_OH * np.sin(angle),
            0
        ])
        # Rotational state: orientation quaternion for each molecule
        self.orientations = np.tile(np.array([1.0,0.0,0.0,0.0], dtype=float), (self.N,1))  # identity quaternion, float
        self.ang_velocities = self.init_angular_velocities()
        self.I = 2.99e-47  # moment of inertia (kg m^2)
        self.minimization_mode = False  # Default value for minimization mode

    def init_angular_velocities(self):
        # Random angular vDDelocities (thermalized)
        I = 2.99e-47  # kg m^2 (approximate for H2O)
        sigma = np.sqrt(1.380649e-23 * self.T / I)
        return np.random.normal(0, sigma, (self.N, 3))

    def init_positions_grid(self):
        # Place molecules on a cubic grid to avoid overlaps
        n = int(np.ceil(self.N ** (1/3)))
        grid = np.linspace(0, self.L, n, endpoint=False)
        mesh = np.array(np.meshgrid(grid, grid, grid)).T.reshape(-1, 3)
        return mesh[:self.N]

    def init_positions(self):
        # Place molecules randomly in the box (could use grid for no overlap)
        return np.random.rand(self.N, 3) * self.L

    def init_velocities(self):
        sigma = np.sqrt(1.380649e-23 * self.T / self.mass)
        return np.random.normal(0, sigma, (self.N, 3))

    def get_atom_positions(self, mol_pos, orientation=None):
        # Returns 3x3 array: [O, H1, H2] positions for a molecule at mol_pos
        if orientation is None:
            rel_H1 = self.rel_H1_0
            rel_H2 = self.rel_H2_0
        else:
            rel_H1 = self.rotate_vector(self.rel_H1_0, orientation)
            rel_H2 = self.rotate_vector(self.rel_H2_0, orientation)
        return np.array([
            mol_pos,  # O
            mol_pos + rel_H1,
            mol_pos + rel_H2
        ])

    @staticmethod
    def rotate_vector(v, q):
        # Rotate vector v by quaternion q
        w, x, y, z = q
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
        ])
        return R @ v

    def potential(self, pos_i, pos_j, ori_i=None, ori_j=None):
        # pos_i, pos_j: center-of-mass positions of two molecules
        atoms_i = self.get_atom_positions(pos_i, ori_i)
        atoms_j = self.get_atom_positions(pos_j, ori_j)
        charges = np.array([-2*self.delta_q, self.delta_q, self.delta_q])
        # Lennard-Jones parameters for each atom: [O, H, H]
        sigmas = np.array([self.sigma_O, self.sigma_H, self.sigma_H])
        epsilons = np.array([self.epsilon_O, self.epsilon_H, self.epsilon_H])
        V = 0.0
        for a in range(3):
            for b in range(3):
                r = np.linalg.norm(atoms_i[a] - atoms_j[b])
                if r == 0:
                    continue
                # Hard-sphere minimum
                if r < self.min_dist:
                    V += 1e5  # Large repulsive energy (J)
                    continue
                # Coulomb
                V += self.k_e * charges[a] * charges[b] / r
                # Lennard-Jones (Lorentz-Berthelot mixing)
                sigma = 0.5 * (sigmas[a] + sigmas[b])
                epsilon = np.sqrt(epsilons[a] * epsilons[b])
                lj = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
                V += lj
        return V

    def forces(self):
        # Compute net force on each molecule (sum of atomic forces)
        F = np.zeros_like(self.positions)
        charges = np.array([-2*self.delta_q, self.delta_q, self.delta_q])
        for i in range(self.N):
            atoms_i = self.get_atom_positions(self.positions[i], self.orientations[i])
            for j in range(self.N):
                if i == j:
                    continue
                atoms_j = self.get_atom_positions(self.positions[j], self.orientations[j])
                for a in range(3):
                    for b in range(3):
                        r_vec = atoms_i[a] - atoms_j[b]
                        r = np.linalg.norm(r_vec)
                        if r == 0:
                            continue
                        f = self.k_e * charges[a] * charges[b] * r_vec / r**3
                        F[i] += f / 3  # distribute atomic force to molecule
        return F

    def update_orientations(self):
        # Integrate angular velocity for each molecule (simple free rotation)
        for i in range(self.N):
            omega = self.ang_velocities[i]
            angle = np.linalg.norm(omega) * self.ts
            if angle == 0:
                continue
            axis = omega / np.linalg.norm(omega)
            q = self.axis_angle_to_quaternion(axis, angle)
            self.orientations[i] = self.quaternion_multiply(q, self.orientations[i])
            self.orientations[i] /= np.linalg.norm(self.orientations[i])
        def update_orientations(self):
            # Langevin thermostat for rotation: friction + noise + update quaternion
            gamma_rot = 1e14  # rotational friction (1/s)
            ts = self.ts
            I = self.I
            T = self.T
            for i in range(self.N):
                omega = self.ang_velocities[i]
                # Friction
                omega *= np.exp(-gamma_rot * ts)
                # Noise (unless minimization mode)
                if not self.minimization_mode:
                    sigma_omega = np.sqrt(2 * gamma_rot * 1.380649e-23 * T / I / ts)
                    omega += np.random.normal(0, sigma_omega, 3)
                self.ang_velocities[i] = omega
                # Integrate orientation
                angle = np.linalg.norm(omega) * ts
                if angle == 0:
                    continue
                axis = omega / np.linalg.norm(omega)
                q = self.axis_angle_to_quaternion(axis, angle)
                self.orientations[i] = self.quaternion_multiply(q, self.orientations[i])
                self.orientations[i] /= np.linalg.norm(self.orientations[i])

    @staticmethod
    def axis_angle_to_quaternion(axis, angle):
        w = np.cos(angle/2)
        xyz = axis * np.sin(angle/2)
        return np.concatenate(([w], xyz))

    @staticmethod
    def quaternion_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def apply_boundary(self):
        self.positions %= self.L

    def step(self):
        # Velocity-Verlet with Langevin thermostat, and rotational update
        F = self.forces()
        gamma = 1e14  # higher friction for stability
        sigma_v = np.sqrt(2 * gamma * 1.380649e-23 * self.T / self.mass / self.ts)
        self.positions += self.velocities * self.ts + 0.5 * F/self.mass * self.ts**2
        F_new = self.forces()
        self.velocities += 0.5 * (F + F_new)/self.mass * self.ts
        self.velocities *= np.exp(-gamma * self.ts)
        self.velocities += np.random.normal(0, sigma_v, self.velocities.shape)
        self.apply_boundary()
        self.update_orientations()
        def step(self):
            # Velocity-Verlet with Langevin thermostat (and minimization mode), plus rotational thermostat
            F = self.forces()
            gamma = 1e15  # much higher friction for stability
            ts = self.ts
            mass = self.mass
            T = self.T
            # Translation: friction
            self.velocities *= np.exp(-gamma * ts)
            # Translation: noise (unless minimization mode)
            if not self.minimization_mode:
                sigma_v = np.sqrt(2 * gamma * 1.380649e-23 * T / mass / ts)
                self.velocities += np.random.normal(0, sigma_v, self.velocities.shape)
            # Velocity-Verlet position update
            self.positions += self.velocities * ts + 0.5 * F/mass * ts**2
            F_new = self.forces()
            self.velocities += 0.5 * (F + F_new)/mass * ts
            self.apply_boundary()
            self.update_orientations()
            # Diagnostic: print minimum interatomic distance (excluding intra-molecular)
            min_dist = np.inf
            for i in range(self.N):
                atoms_i = self.get_atom_positions(self.positions[i], self.orientations[i])
                for j in range(i+1, self.N):
                    atoms_j = self.get_atom_positions(self.positions[j], self.orientations[j])
                    for a in range(3):
                        for b in range(3):
                            r = np.linalg.norm(atoms_i[a] - atoms_j[b])
                            if r < min_dist:
                                min_dist = r
            print(f"[MD] Step: min interatomic distance = {min_dist*1e10:.2f} Å")
