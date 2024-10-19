import numpy as np
import torch
import torch.nn as nn


class MPM(nn.Module):
    def __init__(self, cfg):
        super(MPM, self).__init__()
        self.E = nn.Parameter(torch.ones(1) * cfg.E, requires_grad=True)
        self.cfg = cfg
        self.get_grid_pos()
        self.center = torch.FloatTensor(cfg.center).cuda()

    def set_input(self, data):
        self.pos_seq = []
        self.vel_seq = []
        self.C_seq = []
        self.J_seq = []
        self.vel_field_seq = []  # predicted velocity field
        self.ext_force_seq = []  # exteral force

        pos = data[0].squeeze().cuda()
        vel = data[1].squeeze().cuda()
        C = data[2].squeeze().cuda()
        J = data[3].squeeze().cuda()
        # self.d_vel = torch.norm(vel, p=2, dim=-1)

        self.pos_seq.append(pos)
        self.vel_seq.append(vel)
        self.C_seq.append(C)
        self.J_seq.append(J)

        self.n_particles = pos.size(0)

    def forward(self):
        for s in range(self.n_substeps):  #
            self.substep(s)
            temp = self.grid_v_out.view(self.cfg.n_grid[0], self.cfg.n_grid[1], 2)
            self.vel_field_seq.append(temp)
        return torch.stack(self.vel_field_seq)

    def substep(self, s):
        # get the pos, vel, C, and J of the current step
        self.pos_curr = self.pos_seq[s]
        self.vel_curr = self.vel_seq[s]
        self.C_curr = self.C_seq[s]
        self.J_curr = self.J_seq[s]

        self.init_grid_curr()
        self.init_particle_next()
        self.cal_external_force()
        self.ext_force_seq.append(self.ext_force)

        self.P2G()
        self.OP()
        self.G2P()
        self.apply_boundary()

        self.pos_seq.append(self.pos_next)
        self.vel_seq.append(self.vel_next)
        self.C_seq.append(self.C_next)
        self.J_seq.append(self.J_next)

    def cal_external_force(self):
        dt = self.cfg.dt
        p_mass = self.cfg.p_mass
        r_vec = self.pos_curr - self.center
        r = torch.norm(r_vec, p=2, dim=-1)
        vel_r = torch.mul(torch.div(torch.sum(torch.mul(r_vec, self.vel_curr), dim=1), r ** 2).unsqueeze(1), r_vec)
        vel_t = self.vel_curr - vel_r
        v_t = torch.norm(vel_t, p=2, dim=-1)
        # centripetal force
        temp = p_mass * v_t ** 2 / r
        force = -1 * dt * torch.mul(temp.unsqueeze(1), r_vec / r.unsqueeze(1))
        self.ext_force = force

    def window_fn(self, dis, dx):
        w = torch.zeros(dis.size()).cuda()
        new_dis = torch.abs(dis / dx)

        ## dis < 0.5
        mask0 = torch.logical_and(new_dis >= 0, new_dis < 0.5)
        w[mask0] = 0.75 - new_dis[mask0].pow(2)

        ## dis >= 0.5 and dis < 1.5
        mask1 = torch.logical_and(new_dis >= 0.5, new_dis < 1.5)
        w[mask1] = 0.5 * (1.5 - new_dis[mask1]).pow(2)
        return w

    def P2G(self, ):
        p_mass = self.cfg.p_mass
        n_grids = self.cfg.n_grid[0] * self.cfg.n_grid[1]
        dx = self.cfg.dx
        inv_dx = self.cfg.inv_dx
        dt = self.cfg.dt
        p_vol = self.cfg.p_vol

        dis_vec = self.grid_pos.unsqueeze(1) - self.pos_curr.unsqueeze(0)
        weight_vec = self.window_fn(dis_vec, dx)
        weight = torch.mul(weight_vec[:, :, 0], weight_vec[:, :, 1])

        stress = - dt * 4 * inv_dx ** 2 * self.E * p_vol * (self.J_curr - 1)
        identity_matrix = torch.eye(2).unsqueeze(0).cuda()
        stress_tensor = torch.mul(identity_matrix, stress.unsqueeze(1).unsqueeze(1))

        momentum = (p_mass * self.vel_curr).unsqueeze(0).repeat(n_grids, 1, 1)
        angle_momentum = torch.matmul(p_mass * self.C_curr, dis_vec.unsqueeze(3)).squeeze(-1)
        elastic_force = torch.matmul(stress_tensor, dis_vec.unsqueeze(3)).squeeze(-1)
        external_force = self.ext_force.unsqueeze(0).repeat(n_grids, 1, 1)

        w_momentum = torch.mul(weight.unsqueeze(-1), momentum)
        w_angle_momentum = torch.mul(weight.unsqueeze(-1), angle_momentum)
        w_elastic_force = torch.mul(weight.unsqueeze(-1), elastic_force)
        w_external_force = torch.mul(weight.unsqueeze(-1), external_force)
        total_momentum = w_momentum + w_angle_momentum + w_elastic_force + w_external_force

        self.grid_v_in = total_momentum.sum(1)
        self.grid_mass = (weight * p_mass).sum(1)

    def OP(self):
        mask = self.grid_mass > 0
        self.grid_v_out[mask, :] = torch.div(self.grid_v_in[mask, :], self.grid_mass[mask].unsqueeze(-1))

    def G2P(self):
        dx = self.cfg.dx
        dt = self.cfg.dt
        inv_dx = self.cfg.inv_dx

        dis_vec = self.grid_pos.unsqueeze(1) - self.pos_curr.unsqueeze(0)
        weight_vec = self.window_fn(dis_vec, dx)
        weight = torch.mul(weight_vec[:, :, 0], weight_vec[:, :, 1])

        self.vel_next = torch.matmul(weight.t(), self.grid_v_out)
        # self.vel_next = torch.matmul(weight.t().cpu(), self.grid_v_out.cpu()).cuda()
        out_product = torch.matmul(self.grid_v_out.unsqueeze(-1), dis_vec.permute(1, 0, 2).unsqueeze(-2))
        weight_out_product = torch.mul(weight.t().unsqueeze(2).unsqueeze(2), out_product) * 4 * inv_dx ** 2
        self.C_next = weight_out_product.sum(1)
        for i in range(self.n_particles):
            self.J_next[i] = self.J_curr[i] * (1 + dt * self.C_next[i, :, :].trace())
        self.pos_next = self.pos_curr + dt * self.vel_next

    def apply_boundary(self):
        res = self.cfg.res
        b_radius = self.cfg.b_radius
        dx = self.cfg.dx
        alpha = -0.1

        # bottom boundary
        b_bound = 2 * dx + b_radius
        mask_b = torch.logical_and(self.pos_next[:, 1] < b_bound, self.vel_next[:, 1] < 0)
        self.pos_next[mask_b, 1] = b_bound
        self.vel_next[mask_b, 1] *= alpha

        # top boundary
        t_bound = res[1] - 4 * dx - b_radius
        mask_t = torch.logical_and(self.pos_next[:, 1] > t_bound, self.vel_next[:, 1] > 0)
        self.pos_next[mask_t, 1] = t_bound
        self.vel_next[mask_t, 1] *= alpha

        # right boundary
        r_bound = res[0] - 2 * dx - b_radius
        mask_r = torch.logical_and(self.pos_next[:, 0] > r_bound, self.vel_next[:, 0] > 0)
        self.pos_next[mask_r, 0] = r_bound
        self.vel_next[mask_r, 0] *= alpha

        # left boundary
        l_bound = 2 * dx + b_radius
        mask_l = torch.logical_and(self.pos_next[:, 0] < l_bound, self.vel_next[:, 0] < 0)
        self.pos_next[mask_l, 0] = l_bound
        self.vel_next[mask_l, 0] *= alpha

        # inner bottom boundary
        in_b_bound = 140 - b_radius
        in_mask_b_1 = torch.logical_and(self.pos_next[:, 1] > in_b_bound, self.pos_next[:, 1] <= 160)
        in_mask_b_2 = torch.logical_and(self.pos_next[:, 0] >= 280, self.pos_next[:, 0] <= 320)
        in_mask_b = torch.logical_and(torch.logical_and(in_mask_b_1, in_mask_b_2), self.vel_next[:, 1] > 0)
        self.pos_next[in_mask_b, 1] = in_b_bound
        self.vel_next[in_mask_b, 1] *= alpha

        # top boundary
        in_t_bound = 180 + b_radius
        in_mask_t_1 = torch.logical_and(self.pos_next[:, 1] > 160, self.pos_next[:, 1] < in_t_bound)
        in_mask_t_2 = torch.logical_and(self.pos_next[:, 0] >= 280, self.pos_next[:, 0] <= 320)
        in_mask_t = torch.logical_and(torch.logical_and(in_mask_t_1, in_mask_t_2), self.vel_next[:, 1] < 0)
        self.pos_next[in_mask_t, 1] = in_t_bound
        self.vel_next[in_mask_t, 1] *= alpha

        # right boundary
        in_r_bound = 320 + b_radius
        in_mask_r_1 = torch.logical_and(self.pos_next[:, 0] >= 300, self.pos_next[:, 0] < in_r_bound)
        in_mask_r_2 = torch.logical_and(self.pos_next[:, 1] >= 140, self.pos_next[:, 1] <= 180)
        in_mask_r = torch.logical_and(torch.logical_and(in_mask_r_1, in_mask_r_2), self.vel_next[:, 0] < 0)
        self.pos_next[in_mask_r, 0] = in_r_bound
        self.vel_next[in_mask_r, 0] *= alpha

        # left boundary
        in_l_bound = 280 - b_radius
        in_mask_l_1 = torch.logical_and(self.pos_next[:, 0] > in_l_bound, self.pos_next[:, 0] < 300)
        in_mask_l_2 = torch.logical_and(self.pos_next[:, 1] >= 140, self.pos_next[:, 1] <= 180)
        in_mask_l = torch.logical_and(torch.logical_and(in_mask_l_1, in_mask_l_2), self.vel_next[:, 0] > 0)
        self.pos_next[in_mask_l, 0] = in_l_bound
        self.vel_next[in_mask_l, 0] *= alpha

    def init_grid_curr(self):
        n_grid = self.cfg.n_grid
        dim = self.cfg.dim
        grid_num = n_grid[0] * n_grid[1]
        self.grid_v_in = torch.zeros(grid_num, dim).cuda()
        self.grid_v_out = torch.zeros(grid_num, dim).cuda()
        self.grid_mass = torch.zeros(grid_num).cuda()

    def init_particle_next(self):
        dim = self.cfg.dim
        n_particles = self.n_particles
        self.pos_next = torch.zeros(n_particles, dim).cuda()
        self.vel_next = torch.zeros(n_particles, dim).cuda()
        self.C_next = torch.zeros(n_particles, dim, dim).cuda()
        self.J_next = torch.zeros(n_particles).cuda()

    def get_grid_pos(self):
        coor_x = torch.arange(self.cfg.n_grid[0])
        coor_y = torch.arange(self.cfg.n_grid[1])
        grid_pos_x, grid_pos_y = torch.meshgrid(coor_x, coor_y)
        grid_pos = torch.stack([grid_pos_x, grid_pos_y], dim=2).cuda()
        self.grid_pos = grid_pos.view(-1, 2) * self.cfg.dx
        self.grid_pos.requires_grad = False
