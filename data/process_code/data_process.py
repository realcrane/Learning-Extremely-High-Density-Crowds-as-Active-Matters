import scipy.io as sio
import taichi as ti
from config.config import *


@ti.data_oriented
class DataProcess:
    def __init__(self, ):
        self.pixel_vel = ti.Vector.field(dim, dtype=ti.f32, shape=(pixel_res[0], pixel_res[1]))
        self.vel_field = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid[0], n_grid[1]))
        self.mass_field = ti.field(dtype=ti.f32, shape=(n_grid[0], n_grid[1]))

    def obtain_field(self, id, data_path):
        self.reset_field()

        #### load data
        pixel_vel = self.load_optical_flow(id, data_path)
        pixel_vel = pixel_vel / scale
        self.pixel_vel.from_numpy(pixel_vel)
        self.pixel_to_field(self.pixel_vel, self.vel_field, self.mass_field)
        self.apply_normalize(self.vel_field, self.mass_field)

    @ti.kernel
    def pixel_to_field(self, vel_tensor: ti.template(), vel_field: ti.template(), mass_field: ti.template()):
        for p in ti.grouped(ti.ndrange(pixel_res[0], pixel_res[1])):
            if self.is_inside(p):
                Xp = p / scale * inv_dx
                base = ti.cast(Xp - 0.5, ti.i32)
                fx = Xp - base
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # Bspline

                for i, j in ti.static(ti.ndrange(3, 3)):
                    offset = ti.Vector([i, j])
                    weight = w[i].x * w[j].y
                    temp = vel_tensor[p] * p_mass
                    vel_field[base + offset] += weight * temp
                    mass_field[base + offset] += weight * p_mass

    @ti.func
    def is_inside(self, p):
        flag = p[0] >= pixel_bound and p[0] < (pixel_res[0] - pixel_bound) and p[1] >= pixel_bound + 10 and p[1] < (
                pixel_res[1] - pixel_bound)
        return flag

    @ti.kernel
    def apply_normalize(self, vel_field: ti.template(), mass_field: ti.template()):
        for i, j in ti.ndrange(n_grid[0], n_grid[1]):
            if mass_field[i, j] > 0:
                vel_field[i, j] /= mass_field[i, j]

    def load_optical_flow(self, frame, data_path):
        o_flow_x_path = data_path + '/flow_x_{}.mat'.format(str(frame).zfill(5))
        o_flow_x = sio.loadmat(o_flow_x_path, squeeze_me=True, struct_as_record=False)['data']
        o_flow_y_path = data_path + '/flow_y_{}.mat'.format(str(frame).zfill(5))
        o_flow_y = sio.loadmat(o_flow_y_path, squeeze_me=True, struct_as_record=False)['data']
        o_flow_x = o_flow_x * 50
        o_flow_y = o_flow_y * 50
        temp = np.zeros([pixel_res[0], pixel_res[1], 2])
        temp[pixel_bound:pixel_res[0] - pixel_bound, pixel_bound + 10:pixel_res[1] - pixel_bound, 0] = np.transpose(o_flow_x)
        temp[pixel_bound:pixel_res[0] - pixel_bound, pixel_bound + 10:pixel_res[1] - pixel_bound, 1] = np.transpose(o_flow_y)
        return np.array(temp).astype('float32')

    def reset_field(self):
        self.pixel_vel.fill(0.0)
        self.vel_field.fill(0.0)
        self.mass_field.fill(0.0)
