import numpy as np
from ising_grid import generate_ising_grid

#Parameters
temp = 2.269
size = 32
sweeps = 10
samples = 1000

def generate_dataset(num_samples, size, sweeps, beta):

    dataset = np.zeros((num_samples, size, size), dtype=np.int8)

    for i in range(num_samples):

        dataset[i] = generate_ising_grid(size, sweeps, beta)

        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{num_samples} samples")

    np.save(f"ising_dataset_{size}x{size}.npy", dataset)

if __name__ == "__main__":
    generate_dataset(samples, size, sweeps, 1/temp)