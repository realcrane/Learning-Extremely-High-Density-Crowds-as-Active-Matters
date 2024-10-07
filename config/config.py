import math
import numpy as np

print('using this configuration')

dim = 2

# configure of grid
scale = 10
dx = 20

bound = 3
pixel_bound = bound * dx
n_grid = [24 + bound * 2, 14 + bound * 2]
res = [n_grid[0] * dx, n_grid[1] * dx]
world_res = np.array(res).astype(int)
inv_dx = 1 / dx

b_radius = dx / 8
comfort_dis = 2 * b_radius
p_radius = b_radius + comfort_dis / 2
theta_1 = 2 * b_radius
theta_2 = comfort_dis + 2 * b_radius


center = [300, 160]

# configure of particles
p_rho = 1
p_vol = math.pi * p_radius ** 2
p_mass = p_rho * p_vol

# other configuration
steps = 1
dt = 0.04

# train_30
num_workers = 4
seed = 1992
method = 'mpm'
