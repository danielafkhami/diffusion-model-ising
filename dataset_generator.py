import numpy as np
from ising_grid import generate_ising_grid

#Parameters
temp = 2.269
size = 32
sweeps = 10
samples = 5000

def generate_dataset(num_samples, size, sweeps, beta):

    np.random.seed(42)
    dataset = np.zeros((num_samples, size, size), dtype=np.int8)

    for i in range(num_samples):

        dataset[i] = generate_ising_grid(size, sweeps, beta)

        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{num_samples} samples")

    np.save(f"ising_dataset_{size}x{size}.npy", dataset)
    print("Saved.")

if __name__ == "__main__":
    generate_dataset(samples, size, sweeps, 1/temp)