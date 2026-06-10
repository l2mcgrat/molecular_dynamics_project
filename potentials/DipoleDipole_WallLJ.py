
import numpy as np
from potentials.DD_and_LJ_waters_in_a_box import DipoleDipole


class DipoleDipoleWallLJ(DipoleDipole):
    """
    Water molecules in a confined box with dipole-dipole interactions and
    Lennard-Jones wall potentials.

    Molecules interact via dipole-dipole (plus LJ repulsive core) with each
    other, and experience a full LJ 12-6 potential from each wall face when
    their centre-of-mass is within wall_cutoff of that face.

    Velocity initialisation uses a Maxwell-Boltzmann distribution (drawn from
    a zero-mean Gaussian with sigma = sqrt(kB*T/m) per component, which is the
    component-wise MB distribution).

    Boundary conditions are reflective rather than periodic, so the box acts as
    a physical container supported by the wall LJ repulsion.
    """

    # Wall LJ parameters (representative O–wall interaction)
    sigma_wall = 3.2e-10    # m  (O–O length scale)
    epsilon_wall = 0.65e-21  # J  (well depth)
    wall_cutoff = 0.5e-9    # 0.5 nm

    def __init__(self, N=8, T=100.0, ts=1e-15,
                 box_length=2e-9, mass=18 * 1.66054e-27):
        super().__init__(N=N, T=T, ts=ts, box_length=box_length, mass=mass)

    # ------------------------------------------------------------------
    # Position initialisation: cubic grid well inside the box
    # ------------------------------------------------------------------

    def init_positions(self):
        """Place molecules on a uniform cubic grid inside the box.

        A margin of 0.30*L from each wall keeps the initial positions
        outside the wall LJ cutoff (0.5 nm for a 2 nm box → 0.30*2nm=0.6nm
        margin, so first grid point is 0.6nm from the wall).
        """
        n = int(np.ceil(self.N ** (1.0 / 3.0)))
        margin = 0.30 * self.L
        grid = np.linspace(margin, self.L - margin, n)
        mesh = np.array(np.meshgrid(grid, grid, grid)).T.reshape(-1, 3)
        return mesh[: self.N]

    # ------------------------------------------------------------------
    # Wall LJ potential and forces
    # ------------------------------------------------------------------

    def wall_potential_single(self, pos):
        """LJ 12-6 wall potential for a single molecule at *pos*."""
        V = 0.0
        for dim in range(3):
            for d in (pos[dim], self.L - pos[dim]):
                if 0.0 < d < self.wall_cutoff:
                    r6 = (self.sigma_wall / d) ** 6
                    V += 4.0 * self.epsilon_wall * (r6 * r6 - r6)
        return V

    def wall_forces(self):
        """Forces on all molecules from the LJ walls."""
        F = np.zeros_like(self.positions)
        for i in range(self.N):
            for dim in range(3):
                # --- near the 0-wall (face at dim=0) ---
                d0 = self.positions[i, dim]
                if 0.0 < d0 < self.wall_cutoff:
                    r6 = (self.sigma_wall / d0) ** 6
                    # -dV/d(pos[dim]) pushes molecule in +dim direction
                    F[i, dim] += 4.0 * self.epsilon_wall * (12.0 * r6 * r6 - 6.0 * r6) / d0

                # --- near the L-wall (face at dim=L) ---
                dL = self.L - self.positions[i, dim]
                if 0.0 < dL < self.wall_cutoff:
                    r6 = (self.sigma_wall / dL) ** 6
                    # pushes molecule in -dim direction
                    F[i, dim] -= 4.0 * self.epsilon_wall * (12.0 * r6 * r6 - 6.0 * r6) / dL

        return F

    # ------------------------------------------------------------------
    # Override forces to include wall contribution
    # ------------------------------------------------------------------

    def forces(self):
        return super().forces() + self.wall_forces()

    # ------------------------------------------------------------------
    # Reflective boundary conditions (replaces periodic wrapping)
    # ------------------------------------------------------------------

    def apply_boundary(self):
        """Elastic reflection off all six walls."""
        for dim in range(3):
            # Reflect off the 0-wall
            mask0 = self.positions[:, dim] < 0.0
            self.positions[mask0, dim] = -self.positions[mask0, dim]
            self.velocities[mask0, dim] = np.abs(self.velocities[mask0, dim])

            # Reflect off the L-wall
            maskL = self.positions[:, dim] > self.L
            self.positions[maskL, dim] = 2.0 * self.L - self.positions[maskL, dim]
            self.velocities[maskL, dim] = -np.abs(self.velocities[maskL, dim])

    # ------------------------------------------------------------------
    # Override step() with a correctly calibrated Langevin thermostat.
    #
    # The parent class has sigma_v = sqrt(2*gamma*kB*T / mass / ts), which
    # places ts in the denominator and produces non-physical velocities.
    # The correct fluctuation-dissipation relation for a discrete Langevin
    # step of length dt is:
    #
    #   v_new = v * exp(-gamma*dt)
    #           + sqrt(kB*T/m * (1 - exp(-2*gamma*dt))) * xi
    #
    # For gamma*dt = 1e13 * 1e-15 = 0.01 this gives sigma_v ≈ 30 m/s,
    # consistent with the 100 K thermal velocity scale (~214 m/s rms).
    # ------------------------------------------------------------------

    def step(self):
        # Velocity-Verlet integration
        F = self.forces()
        self.positions += self.velocities * self.ts + 0.5 * F / self.mass * self.ts**2
        F_new = self.forces()
        self.velocities += 0.5 * (F + F_new) / self.mass * self.ts
        self.apply_boundary()

        # Langevin thermostat (exact fluctuation-dissipation)
        gamma = 1e13  # friction coefficient (s^-1); gamma*ts = 0.01 << 1
        alpha = np.exp(-gamma * self.ts)
        sigma_v = np.sqrt(self.kB * self.T / self.mass * (1.0 - alpha ** 2))
        self.velocities = (self.velocities * alpha
                           + np.random.normal(0.0, sigma_v, self.velocities.shape))

        # Update dipole orientations
        self.update_dipole_orientations()
