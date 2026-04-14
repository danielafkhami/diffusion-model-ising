import matplotlib.pyplot as plt
from ising_grid import generate_ising_grid

# Parameters
temp = 2.269
size = 32
sweeps = 50

sample_grid = generate_ising_grid(size, sweeps, 1.0/temp)

plt.imshow(sample_grid, cmap='RdBu')
plt.show()