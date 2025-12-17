
import numpy as np

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
