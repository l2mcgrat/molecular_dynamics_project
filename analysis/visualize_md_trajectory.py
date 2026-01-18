import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# --- Path handling: always relative to project root ---
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# --- User parameters ---
trajectory = "traj_2026-01-17_N16_steps500_T150K.npy"
trajectory_file = os.path.join(PROJECT_ROOT, 'trajectories', trajectory)
orientations_file = None  # Set to .npy file if you have saved orientations, else None
plot_subset = None  # e.g., 20 to plot only 20 molecules, or None for all

# --- Output directory and naming ---
output_dir = os.path.join(SCRIPT_DIR, 'dynamics_videos', 'Molecular_Couloumb_Potential')
os.makedirs(output_dir, exist_ok=True)

# --- Extract N and density from trajectory file or user input ---
import re
N_match = re.search(r'_N(\d+)', trajectory_file)
N = int(N_match.group(1)) if N_match else 0
T_match = re.search(r'_T(\d+)K', trajectory_file)
T = int(T_match.group(1)) if T_match else 0
try:
    density = 777.1  # Set manually or load from file if available
except Exception:
    density = 777.1

output_base = f"md_N{N}_T{T}K_density{density:.1f}"
output_video = os.path.join(output_dir, output_base + '_3d.mp4')
output_video_2d = os.path.join(output_dir, output_base + '_2d.mp4')

# --- Auto-detect ffmpeg, fall back to GIF if needed ---
import matplotlib.animation as animation
import shutil
def get_writer_and_ext(preferred_ext='.mp4', fallback_ext='.gif', fps=20):
    # Use shutil.which to check for ffmpeg in PATH
    if shutil.which('ffmpeg'):
        writer = animation.FFMpegWriter(fps=fps)
        return writer, preferred_ext
    else:
        print("[Warning] ffmpeg not available. Falling back to GIF output.")
        writer = animation.PillowWriter(fps=fps)
        return writer, fallback_ext

writer3d, ext3d = get_writer_and_ext('.mp4', '.gif', fps=20)
writer2d, ext2d = get_writer_and_ext('.mp4', '.gif', fps=20)
output_video = output_video[:-4] + ext3d
output_video_2d = output_video_2d[:-4] + ext2d
plot_subset = None  # e.g., 20 to plot only 20 molecules, or None for all

# --- Water geometry ---
angle = np.deg2rad(104.52)
r_OH = 0.9572e-10  # meters
rel_H1_0 = np.array([r_OH, 0, 0])
rel_H2_0 = np.array([
    r_OH * np.cos(angle),
    r_OH * np.sin(angle),
    0
])

# --- Load trajectory ---
traj = np.load(trajectory_file)  # shape: (timesteps, N, 3)
num_frames, N, _ = traj.shape

# --- Load orientations if available ---
if orientations_file and os.path.exists(orientations_file):
    orientations = np.load(orientations_file)  # shape: (timesteps, N, 4)
else:
    orientations = None

if plot_subset is not None:
    N = min(N, plot_subset)
    traj = traj[:, :N, :]
    if orientations is not None:
        orientations = orientations[:, :N, :]

# --- Helper: rotate vector by quaternion ---
def rotate_vector(v, q):
    w, x, y, z = q
    R = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
    ])
    return R @ v

# --- 3D Animation with O-H bonds ---
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 1e-9])
ax.set_ylim([0, 1e-9])
ax.set_zlim([0, 1e-9])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')

# Oxygens: red, Hydrogens: cyan
scat_O = ax.scatter([], [], [], s=80, c='red', alpha=0.8, label='O')
scat_H = ax.scatter([], [], [], s=40, c='cyan', alpha=0.8, label='H')
bond_lines = [ax.plot([], [], [], 'gray', lw=1.5, alpha=0.5)[0] for _ in range(N*2)]  # 2 bonds per molecule

def update_3d(frame):
    pos = traj[frame]
    O_pos = pos
    H1_pos = np.zeros_like(pos)
    H2_pos = np.zeros_like(pos)
    if orientations is not None:
        for i in range(N):
            q = orientations[frame, i]
            O = pos[i]
            H1 = O + rotate_vector(rel_H1_0, q)
            H2 = O + rotate_vector(rel_H2_0, q)
            H1_pos[i] = H1
            H2_pos[i] = H2
            bond_lines[2*i].set_data([O[0], H1[0]], [O[1], H1[1]])
            bond_lines[2*i].set_3d_properties([O[2], H1[2]])
            bond_lines[2*i+1].set_data([O[0], H2[0]], [O[1], H2[1]])
            bond_lines[2*i+1].set_3d_properties([O[2], H2[2]])
    else:
        H1_pos[:] = pos
        H2_pos[:] = pos
    scat_O._offsets3d = (O_pos[:,0], O_pos[:,1], O_pos[:,2])
    scat_H._offsets3d = (np.concatenate([H1_pos[:,0], H2_pos[:,0]]),
                         np.concatenate([H1_pos[:,1], H2_pos[:,1]]),
                         np.concatenate([H1_pos[:,2], H2_pos[:,2]]))
    return [scat_O, scat_H] + bond_lines

ani3d = FuncAnimation(fig, update_3d, frames=num_frames, blit=False)
ani3d.save(output_video, writer=writer3d)
plt.close(fig)
print(f"Saved 3D animation to {output_video}")

# --- 2D Animation (x vs y) ---
fig2d, ax2d = plt.subplots(figsize=(7,7))
ax2d.set_xlim([0, 1e-9])
ax2d.set_ylim([0, 1e-9])
ax2d.set_xlabel('x (m)')
ax2d.set_ylabel('y (m)')
scat2d_O = ax2d.scatter([], [], s=80, c='red', alpha=0.8, label='O')
scat2d_H = ax2d.scatter([], [], s=40, c='cyan', alpha=0.8, label='H')
bond_lines_2d = [ax2d.plot([], [], 'gray', lw=1.5, alpha=0.5)[0] for _ in range(N*2)]

def update_2d(frame):
    pos = traj[frame]
    O_pos = pos
    H1_pos = np.zeros_like(pos)
    H2_pos = np.zeros_like(pos)
    if orientations is not None:
        for i in range(N):
            q = orientations[frame, i]
            O = pos[i]
            H1 = O + rotate_vector(rel_H1_0, q)
            H2 = O + rotate_vector(rel_H2_0, q)
            H1_pos[i] = H1
            H2_pos[i] = H2
            bond_lines_2d[2*i].set_data([O[0], H1[0]], [O[1], H1[1]])
            bond_lines_2d[2*i+1].set_data([O[0], H2[0]], [O[1], H2[1]])
    else:
        H1_pos[:] = pos
        H2_pos[:] = pos
    scat2d_O.set_offsets(O_pos[:, :2])
    scat2d_H.set_offsets(np.vstack([H1_pos[:, :2], H2_pos[:, :2]]))
    return [scat2d_O, scat2d_H] + bond_lines_2d

ani2d = FuncAnimation(fig2d, update_2d, frames=num_frames, blit=False)
fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, 1e-9])
ax.set_ylim([0, 1e-9])
ax.set_zlim([0, 1e-9])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')

# Oxygens: red, Hydrogens: cyan
scat_O = ax.scatter([], [], [], s=80, c='red', alpha=0.8, label='O')
scat_H = ax.scatter([], [], [], s=40, c='cyan', alpha=0.8, label='H')
bond_lines = [ax.plot([], [], [], 'gray', lw=1.5, alpha=0.5)[0] for _ in range(N*2)]  # 2 bonds per molecule
