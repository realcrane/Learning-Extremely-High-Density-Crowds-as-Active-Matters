import torch
import torch.nn as nn
from model.MyParaNetwork_E import Parameter_Estimate as Parameter_Estimate_E
from model.MyParaNetwork_K import Parameter_Estimate as Parameter_Estimate_K
from model.MyParaNetwork_alpha import Parameter_Estimate as Parameter_Estimate_alpha


class MPM(nn.Module):
    def __init__(self, cfg):
        super(MPM, self).__init__()
        self.cfg = cfg
        self.get_grid_pos()
        self.get_super_paras_alpha = Parameter_Estimate_alpha(in_channels=2, img_size=[30, 20])
        self.get_super_paras_E = Parameter_Estimate_E(particle_radius=cfg.p_radius)
        self.get_super_paras_K = Parameter_Estimate_K(particle_radius=cfg.p_radius)
        self.center = torch.FloatTensor(cfg.center).cuda()

    def set_input(self, data):
        self.E_seq = []
        self.pos_seq = []
        self.vel_seq = []
        self.C_seq = []
        self.J_seq = []
        self.grid_v_out_seq = []
        self.grid_v_in_seq = []
        self.grid_m_seq = []
        self.alpha_seq = []
        self.d_force_seq = []  # impulsive force defined by myself
        self.ext_force_seq = []  # goal attraction

        pos = data[0].squeeze().cuda()
        vel = data[1].squeeze().cuda()
        C = data[2].squeeze().cuda()
        J = data[3].squeeze().cuda()

        self.pos_seq.append(pos)
        self.vel_seq.append(vel)
        self.C_seq.append(C)
        self.J_seq.append(J)

        self.n_particles = pos.size(0)

    def forward(self):
        for s in range(self.n_substeps):  #
            self.substep(s)
            out_temp = self.grid_v_out.view(self.cfg.n_grid[0], self.cfg.n_grid[1], 2)
            in_temp = self.grid_v_in.view(self.cfg.n_grid[0], self.cfg.n_grid[1], 2)
            m_temp = self.grid_mass.view(self.cfg.n_grid[0], self.cfg.n_grid[1], 1)
            alpha_temp = self.alpha.view(self.cfg.n_grid[0], self.cfg.n_grid[1], 1)
            self.grid_v_out_seq.append(out_temp)
            self.grid_v_in_seq.append(in_temp)
            self.grid_m_seq.append(m_temp)
            self.alpha_seq.append(alpha_temp)
        return torch.stack(self.grid_v_out_seq)

    def substep(self, s):
        # get the pos, vel, C, and J of the current step
        self.pos_curr = self.pos_seq[s]
        self.vel_curr = self.vel_seq[s]
        self.C_curr = self.C_seq[s]
        self.J_curr = self.J_seq[s]

        self.E = torch.zeros(self.n_particles, 1).cuda()
        self.K = torch.zeros(self.n_particles, 1).cuda()

        para_net_input = [self.pos_curr, self.vel_curr]
        self.K = (self.get_super_paras_K(para_net_input) + 1) * self.cfg.K
        self.E = (self.get_super_paras_E(para_net_input) + 1) * self.cfg.E

        self.init_grid_curr()
        self.init_particle_next()

        self.cal_defined_force()
        self.cal_external_force()
        self.d_force_seq.append(self.d_force)
        self.ext_force_seq.append(self.ext_force)

        self.P2G()
        self.OP()
        self.G2P()
        self.apply_boundary()

        self.pos_seq.append(self.pos_next)
        self.vel_seq.append(self.vel_next)
        self.C_seq.append(self.C_next)
        self.J_seq.append(self.J_next)

    def cal_defined_force(self, ):
        n_particles = self.n_particles
        comfort_dis = self.cfg.comfort_dis
        theta_1 = self.cfg.theta_1
        theta_2 = self.cfg.theta_2

        force = torch.zeros([n_particles, n_particles, self.cfg.dim]).cuda()
        dis_vec = self.pos_curr.unsqueeze(1) - self.pos_curr.unsqueeze(0)  # p - p'=> p'p
        dis = torch.norm(dis_vec, p=2, dim=-1)
        K = self.K.repeat(1, n_particles)

        mask = torch.logical_and(dis < theta_2, dis > 0)

        if mask.sum() > 0:
            maskd_dis_vec = dis_vec[mask, :]
            masked_dis = dis[mask]
            masked_K = K[mask]

            direction = torch.div(maskd_dis_vec, masked_dis.unsqueeze(1))
            theta = torch.FloatTensor([theta_1 + 1e-5]).cuda()
            new_masked_dis = masked_dis.clone()
            new_masked_dis[masked_dis < theta] = theta

            temp = torch.mul(masked_K, -torch.log((new_masked_dis - theta_1) / comfort_dis) * self.cfg.dt)
            force[mask, :] = torch.mul(temp.unsqueeze(1), direction)
        self.d_force = torch.sum(force, dim=1)

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

        stress = - dt * 4 * self.E.squeeze(1) * inv_dx ** 2 * p_vol * (self.J_curr - 1)
        identity_matrix = torch.eye(2).unsqueeze(0).cuda()
        stress_tensor = torch.mul(identity_matrix, stress.unsqueeze(1).unsqueeze(1))

        momentum = (p_mass * self.vel_curr).unsqueeze(0).repeat(n_grids, 1, 1)
        angle_momentum = torch.matmul(p_mass * self.C_curr, dis_vec.unsqueeze(3)).squeeze(-1)
        elastic_force = torch.matmul(stress_tensor, dis_vec.unsqueeze(3)).squeeze(-1)
        external_force = self.ext_force.unsqueeze(0).repeat(n_grids, 1, 1)
        defined_force = self.d_force.unsqueeze(0).repeat(n_grids, 1, 1)

        w_momentum = torch.mul(weight.unsqueeze(-1), momentum).sum(1)
        w_angle_momentum = torch.mul(weight.unsqueeze(-1), angle_momentum).sum(1)
        w_elastic_force = torch.mul(weight.unsqueeze(-1), elastic_force).sum(1)
        w_external_force = torch.mul(weight.unsqueeze(-1), external_force).sum(1)
        w_defined_force = torch.mul(weight.unsqueeze(-1), defined_force).sum(1)

        self.grid_mass = (weight * p_mass).sum(1)
        mask = self.grid_mass > 0
        momentum_temp = w_momentum + w_angle_momentum
        self.grid_v_in[mask, :] = torch.div(momentum_temp[mask, :], self.grid_mass[mask].unsqueeze(-1))

        grid_v_temp = self.grid_v_in.view(self.cfg.n_grid[0], self.cfg.n_grid[1], 2)
        self.alpha = torch.zeros(n_grids, 1).cuda()
        self.alpha[mask, :] = self.get_super_paras_alpha(grid_v_temp)[mask, :]
        w_active_force_1 = momentum_temp * self.alpha * dt

        self.grid_momentum = w_momentum + w_angle_momentum + w_elastic_force + w_external_force + w_defined_force + w_active_force_1

    def OP(self):
        mask = self.grid_mass > 0
        self.grid_v_out[mask, :] = torch.div(self.grid_momentum[mask, :], self.grid_mass[mask].unsqueeze(-1))

    def G2P(self):
        dx = self.cfg.dx
        dt = self.cfg.dt
        inv_dx = self.cfg.inv_dx

        dis_vec = self.grid_pos.unsqueeze(1) - self.pos_curr.unsqueeze(0)
        weight_vec = self.window_fn(dis_vec, dx)
        weight = torch.mul(weight_vec[:, :, 0], weight_vec[:, :, 1])

        self.vel_next = torch.matmul(weight.t(), self.grid_v_out)
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
        gamma = -0.1

        # bottom boundary
        b_bound = 2 * dx + b_radius
        mask_b = torch.logical_and(self.pos_next[:, 1] < b_bound, self.vel_next[:, 1] < 0)
        self.pos_next[mask_b, 1] = b_bound
        self.vel_next[mask_b, 1] *= gamma

        # top boundary
        t_bound = res[1] - 4 * dx - b_radius
        mask_t = torch.logical_and(self.pos_next[:, 1] > t_bound, self.vel_next[:, 1] > 0)
        self.pos_next[mask_t, 1] = t_bound
        self.vel_next[mask_t, 1] *= gamma

        # right boundary
        r_bound = res[0] - 2 * dx - b_radius
        mask_r = torch.logical_and(self.pos_next[:, 0] > r_bound, self.vel_next[:, 0] > 0)
        self.pos_next[mask_r, 0] = r_bound
        self.vel_next[mask_r, 0] *= gamma

        # left boundary
        l_bound = 2 * dx + b_radius
        mask_l = torch.logical_and(self.pos_next[:, 0] < l_bound, self.vel_next[:, 0] < 0)
        self.pos_next[mask_l, 0] = l_bound
        self.vel_next[mask_l, 0] *= gamma

        # inner bottom boundary
        in_b_bound = 140 - b_radius
        in_mask_b_1 = torch.logical_and(self.pos_next[:, 1] > in_b_bound, self.pos_next[:, 1] <= 160)
        in_mask_b_2 = torch.logical_and(self.pos_next[:, 0] >= 280, self.pos_next[:, 0] <= 320)
        in_mask_b = torch.logical_and(torch.logical_and(in_mask_b_1, in_mask_b_2), self.vel_next[:, 1] > 0)
        self.pos_next[in_mask_b, 1] = in_b_bound
        self.vel_next[in_mask_b, 1] *= gamma

        # top boundary
        in_t_bound = 180 + b_radius
        in_mask_t_1 = torch.logical_and(self.pos_next[:, 1] > 160, self.pos_next[:, 1] < in_t_bound)
        in_mask_t_2 = torch.logical_and(self.pos_next[:, 0] >= 280, self.pos_next[:, 0] <= 320)
        in_mask_t = torch.logical_and(torch.logical_and(in_mask_t_1, in_mask_t_2), self.vel_next[:, 1] < 0)
        self.pos_next[in_mask_t, 1] = in_t_bound
        self.vel_next[in_mask_t, 1] *= gamma

        # right boundary
        in_r_bound = 320 + b_radius
        in_mask_r_1 = torch.logical_and(self.pos_next[:, 0] >= 300, self.pos_next[:, 0] < in_r_bound)
        in_mask_r_2 = torch.logical_and(self.pos_next[:, 1] >= 140, self.pos_next[:, 1] <= 180)
        in_mask_r = torch.logical_and(torch.logical_and(in_mask_r_1, in_mask_r_2), self.vel_next[:, 0] < 0)
        self.pos_next[in_mask_r, 0] = in_r_bound
        self.vel_next[in_mask_r, 0] *= gamma

        # left boundary
        in_l_bound = 280 - b_radius
        in_mask_l_1 = torch.logical_and(self.pos_next[:, 0] > in_l_bound, self.pos_next[:, 0] < 300)
        in_mask_l_2 = torch.logical_and(self.pos_next[:, 1] >= 140, self.pos_next[:, 1] <= 180)
        in_mask_l = torch.logical_and(torch.logical_and(in_mask_l_1, in_mask_l_2), self.vel_next[:, 0] > 0)
        self.pos_next[in_mask_l, 0] = in_l_bound
        self.vel_next[in_mask_l, 0] *= gamma

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
