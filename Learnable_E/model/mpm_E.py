from model.ParaNet_E import Parameter_Estimate
import torch.nn as nn
import torch


class MPM(nn.Module):
    def __init__(self, cfg):
        super(MPM, self).__init__()
        self.cfg = cfg
        self.get_grid_pos()
        self.goal = torch.FloatTensor(cfg.goal).cuda()
        self.get_super_paras_E = Parameter_Estimate(particle_radius=cfg.p_radius)

    def set_input(self, data):
        self.all_pos_seq = []
        self.all_vel_seq = []
        self.all_C_seq = []
        self.all_J_seq = []
        self.all_flag_seq = []
        self.vel_field_seq = []  # predicted velocity field

        self.init_pos = data[0].squeeze().cuda()
        self.init_vel = data[1].squeeze().cuda()
        self.init_ind = data[2].squeeze().cuda()
        self.all_d_vel = torch.norm(self.init_vel, dim=-1)
        self.init_C = data[3].squeeze().cuda()
        self.init_J = data[4].squeeze().cuda()

        self.all_n_particles = self.init_pos.size(0)

        flag = torch.zeros(self.all_n_particles, dtype=bool).cuda()
        flag[self.init_ind == 0] = True
        self.all_flag_seq.append(flag)

        pos = torch.zeros(self.init_pos.size()).cuda()
        vel = torch.zeros(self.init_vel.size()).cuda()
        C = torch.zeros(self.init_C.size()).cuda()
        J = torch.zeros(self.init_J.size()).cuda()

        pos[flag, :] = self.init_pos[flag, :]
        vel[flag, :] = self.init_vel[flag, :]
        C[flag, :, :] = self.init_C[flag, :, :]
        J[flag] = self.init_J[flag]

        self.all_pos_seq.append(pos)
        self.all_vel_seq.append(vel)
        self.all_C_seq.append(C)
        self.all_J_seq.append(J)

    def forward(self):
        for s in range(self.n_substeps):  #
            self.substep(s)
            temp = self.grid_v_out.view(self.cfg.n_grid[0], self.cfg.n_grid[1], 2)
            self.vel_field_seq.append(temp)
        return torch.stack(self.vel_field_seq)

    def substep(self, s):
        self.all_flag_curr = self.all_flag_seq[s]

        # get current properties of all particles
        self.all_pos_curr = self.all_pos_seq[s]
        self.all_vel_curr = self.all_vel_seq[s]
        self.all_C_curr = self.all_C_seq[s]
        self.all_J_curr = self.all_J_seq[s]

        # check whether there are new particles entry the view
        if s > 0:
            new_flag = self.init_ind == s
            if new_flag.sum() > 0:
                self.all_flag_curr[new_flag] = True

                # initialize the pos and vel for new particles
                self.all_pos_curr[new_flag, :] = self.init_pos[new_flag, :]
                self.all_vel_curr[new_flag, :] = self.init_vel[new_flag, :]
                self.all_C_curr[new_flag, :, :] = self.init_C[new_flag, :, :]
                self.all_J_curr[new_flag] = self.init_J[new_flag]

        self.all_flag_next = torch.zeros(self.all_n_particles, dtype=bool).cuda()
        self.all_flag_next.copy_(self.all_flag_curr)

        # the number of visible particles
        self.n_particles = self.all_flag_curr.sum()

        # get the current visible particles
        self.pos_curr = self.all_pos_curr[self.all_flag_curr, :]
        self.vel_curr = self.all_vel_curr[self.all_flag_curr, :]
        self.C_curr = self.all_C_curr[self.all_flag_curr, :, :]
        self.J_curr = self.all_J_curr[self.all_flag_curr]
        self.d_vel = self.all_d_vel[self.all_flag_curr]

        assert self.n_particles == self.pos_curr.size(0) == self.vel_curr.size(0) == self.C_curr.size(
            0) == self.J_curr.size(0), 'something wrong'

        # flag for whether a particle reach the goal/ if reach, value equal to False
        self.reach_goal_flag = torch.zeros(self.n_particles, dtype=bool).cuda()

        # estimate E
        self.E = torch.zeros(self.n_particles, 1).cuda()
        para_net_input = [self.pos_curr, self.vel_curr]
        self.E = self.get_super_paras_E(para_net_input) * self.cfg.E

        self.init_grid_curr()
        self.init_all_particle_next()
        self.init_particle_next()
        self.cal_external_force()

        self.P2G()
        self.OP()
        self.G2P()
        self.apply_boundary()

        self.all_flag_next[self.all_flag_curr] = ~self.reach_goal_flag
        self.all_pos_next[self.all_flag_curr, :] = self.pos_next
        self.all_vel_next[self.all_flag_curr, :] = self.vel_next
        self.all_C_next[self.all_flag_curr, :, :] = self.C_next
        self.all_J_next[self.all_flag_curr] = self.J_next

        self.all_pos_seq.append(self.all_pos_next)
        self.all_vel_seq.append(self.all_vel_next)
        self.all_C_seq.append(self.all_C_next)
        self.all_J_seq.append(self.all_J_next)
        self.all_flag_seq.append(self.all_flag_next)

    def cal_external_force(self):
        dt = self.cfg.dt
        p_mass = self.cfg.p_mass
        dis_vec = self.goal - self.pos_curr
        dis = torch.norm(dis_vec, p=2, dim=-1)
        normal = torch.div(dis_vec, dis.unsqueeze(1))
        force = (torch.mul(self.d_vel.unsqueeze(1), normal) - self.vel_curr) / dt * p_mass * self.cfg.w_ext
        self.ext_force = dt * force

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

        stress = - dt * 4 * inv_dx ** 2 * self.E.squeeze(1) * p_vol * (self.J_curr - 1)
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
        out_product = torch.matmul(self.grid_v_out.unsqueeze(-1), dis_vec.permute(1, 0, 2).unsqueeze(-2))
        weight_out_product = torch.mul(weight.t().unsqueeze(2).unsqueeze(2), out_product) * 4 * inv_dx ** 2
        self.C_next = weight_out_product.sum(1)
        for i in range(self.n_particles):
            self.J_next[i] = self.J_curr[i] * (1 + dt * self.C_next[i, :, :].trace())
        self.pos_next = self.pos_curr + dt * self.vel_next

    def apply_boundary(self):
        dx = self.cfg.dx
        res = self.cfg.res
        b_radius = self.cfg.b_radius
        bound = self.cfg.bound
        door_l_pos = self.cfg.door_l_pos
        door_r_pos = self.cfg.door_r_pos
        theta = -0.1

        # reach the goal
        mask1 = torch.logical_and(self.pos_next[:, 0] >= door_l_pos, self.pos_next[:, 0] < door_r_pos)
        mask2 = self.pos_next[:, 1] > (res[1] - bound * dx + dx)
        invis_mask = torch.logical_and(mask1, mask2)
        self.reach_goal_flag[invis_mask] = True

        # bottom boundary
        b_bound = 2 * dx + b_radius
        mask_b = torch.logical_and(self.pos_next[:, 1] < b_bound, self.vel_next[:, 1] < 0)
        self.pos_next[mask_b, 1] = b_bound
        self.vel_next[mask_b, 1] *= theta

        # top boundary
        t_bound = res[1] - bound * dx - b_radius
        mask_t1 = torch.logical_and(self.pos_next[:, 1] > t_bound, self.vel_next[:, 1] > 0)
        mask_t2 = torch.logical_or(self.pos_next[:, 0] < door_l_pos, self.pos_next[:, 0] >= door_r_pos)
        mask_t = torch.logical_and(mask_t1, mask_t2)
        self.pos_next[mask_t, 1] = t_bound
        self.vel_next[mask_t, 1] *= theta

        # right boundary
        r_bound = res[0] - 2 * dx - b_radius
        mask_r = torch.logical_and(self.pos_next[:, 0] > r_bound, self.vel_next[:, 0] > 0)
        self.pos_next[mask_r, 0] = r_bound
        self.vel_next[mask_r, 0] *= theta

        # left boundary
        l_bound = 2 * dx + b_radius
        mask_l = torch.logical_and(self.pos_next[:, 0] < l_bound, self.vel_next[:, 0] < 0)
        self.pos_next[mask_l, 0] = l_bound
        self.vel_next[mask_l, 0] *= theta

    def init_grid_curr(self):
        n_grid = self.cfg.n_grid
        dim = self.cfg.dim
        grid_num = n_grid[0] * n_grid[1]
        self.grid_v_in = torch.zeros(grid_num, dim).cuda()
        self.grid_v_out = torch.zeros(grid_num, dim).cuda()
        self.grid_mass = torch.zeros(grid_num).cuda()

    def init_all_particle_next(self):
        dim = self.cfg.dim
        all_n_particles = self.all_n_particles
        self.all_pos_next = torch.zeros(all_n_particles, dim).cuda()
        self.all_vel_next = torch.zeros(all_n_particles, dim).cuda()
        self.all_C_next = torch.zeros(all_n_particles, dim, dim).cuda()
        self.all_J_next = torch.zeros(all_n_particles).cuda()

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
