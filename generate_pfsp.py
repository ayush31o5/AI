import numpy as np

def generate_random_pfsp(n_jobs, n_machines, low=1, high=100, seed=None):
    """
    Returns an (n_jobs × n_machines) array of integer processing times,
    uniformly sampled between [low, high].
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(low, high+1, size=(n_jobs, n_machines))

# Example: 20 jobs × 5 machines
pt = generate_random_pfsp(20, 5, low=10, high=200, seed=42)

# Save it in the same text format your loader expects:
np.savetxt("random_pfsp_20x5.txt", pt, fmt="%d")
