import math
import numpy as np

print('using this configuration')

dim = 2

# configure of scene
scale = 10
dx = 2

bound = 3
pixel_bound = bound * dx * scale
door_l_pos = 19 + bound * dx
door_r_pos = 27 + bound * dx

# configure of grid

n_grid = [24 + bound * 2, 14 + bound * 2]
res = [n_grid[0] * dx, n_grid[1] * dx]
pixel_res = [n_grid[0] * dx * scale, n_grid[1] * dx * scale]
world_res = np.array(res).astype(int) * 2 * 10
inv_dx = 1 / dx

goal = [23 + bound * dx, 32 + bound * dx]

# configure of particles
b_radius = dx / 2
comfort_dis = 2 * b_radius
p_radius = b_radius + comfort_dis / 2
theta_1 = 2 * b_radius
theta_2 = comfort_dis + 2 * b_radius

p_rho = 1
p_vol = math.pi * p_radius ** 2
p_mass = p_rho * p_vol

# other configuration
steps = 1
dt = 0.02

# train
num_workers = 4
seed = 1992
method = 'mpm'