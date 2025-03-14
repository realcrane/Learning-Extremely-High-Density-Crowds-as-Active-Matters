import numpy as np
import taichi as ti
from config.config import *


@ti.data_oriented
class DataProcess:
    def __init__(self, ):
        self.vel_field = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid[0], n_grid[1]))
        self.pixel_field = ti.Vector.field(2, dtype=ti.f32, shape=(pixel_res[0], pixel_res[1]))

    def obtain_pixel_image_list(self, field_array):
        pixel_image_list = []
        for i in range(field_array.shape[0]):
            self.reset_field()
            self.vel_field.from_numpy(field_array[i, :, :, :])
            self.field_to_pixel(self.vel_field)
            pixel_image = self.pixel_field.to_numpy()
            pixel_image_list.append(pixel_image[60:540, 70:340, :])

        return np.array(pixel_image_list)

    @ti.kernel
    def field_to_pixel(self, vel_field: ti.template()):
        for p in ti.grouped(ti.ndrange(pixel_res[0], pixel_res[1])):
            if self.is_inside(p):
                Xp = p / 10 / dx
                base = ti.cast(Xp - 0.5, ti.i32)
                fx = Xp - base
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]  # Bspline

                new_v = ti.Vector.zero(float, 2)
                for i, j in ti.static(ti.ndrange(3, 3)):
                    offset = ti.Vector([i, j])
                    weight = w[i].x * w[j].y
                    g_v = vel_field[base + offset]
                    new_v += weight * g_v
                self.pixel_field[p] = new_v

    @ti.func
    def is_inside(self, p):
        flag = p[0] >= pixel_bound and p[0] < (pixel_res[0] - pixel_bound) and p[1] >= pixel_bound + 10 and p[1] < (
                pixel_res[1] - pixel_bound)
        return flag

    def reset_field(self):
        self.pixel_field.fill(0.0)
        self.vel_field.fill(0.0)
