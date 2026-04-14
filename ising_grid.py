import numpy as np

def generate_ising_grid(size, sweeps, beta):

    grid = np.random.choice([1, -1], size=(size, size))
    steps = sweeps * (size * size)

    for i in range(steps):

        x = np.random.randint(0, size)
        y = np.random.randint(0, size)

        nb_sum = (grid[(x + 1) % size, y] + grid[(x - 1) % size, y] + grid[x, (y + 1) % size] + grid[x, (y - 1) % size])

        dE = 2 * grid[x, y] * nb_sum

        if dE <= 0 or np.random.rand() < np.exp(-dE*beta):
            grid[x, y] *= -1

    return grid