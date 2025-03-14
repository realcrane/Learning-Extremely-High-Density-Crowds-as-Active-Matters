from utils.utils import Log, seed_everything, count_parameters, Metric
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from dataset.dataset import Dataset
import sys

sys.path.append('../../')
from config import config as cfg
from model.mpm_E import MPM
from time import time
from tqdm import tqdm
import torch
import os

def train(model, optimizer, scheduler, train_loader, cfg, epoch):
    model.train()
    model.n_substeps = cfg.n_substeps
    loss = Metric('mseloss')

    # Start batch training
    infos = 'Train {} epoch {}/{}'.format('MPM only E', epoch, cfg.epochs)
    for init_pos, init_vel, init_ind, vel_field_gt in tqdm(train_loader, desc=infos):
        init_pos = init_pos.squeeze(0)
        init_vel = init_vel.squeeze(0)
        init_ind = init_ind.squeeze(0)
        vel_field_gt = vel_field_gt.squeeze(0).cuda()

        num = init_pos.size(0)
        C = torch.zeros((num, cfg.dim, cfg.dim))
        J = torch.ones((num))

        model.set_input([init_pos, init_vel, init_ind, C, J])
        vel_field_pred = model.forward()
        masked_pred = vel_field_pred[:, cfg.bound:cfg.n_grid[0] - cfg.bound,
                      cfg.bound:cfg.n_grid[1] - cfg.bound, :]
        masked_gt = vel_field_gt[:, cfg.bound:cfg.n_grid[0] - cfg.bound,
                    cfg.bound:cfg.n_grid[1] - cfg.bound, :]
        mseLoss = mse_loss(masked_pred, masked_gt)
        loss.update(mseLoss.detach().cpu(), 1)
        optimizer.zero_grad()
        mseLoss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
    scheduler.step()
    return loss.avg


def val(model, val_loader, cfg, epoch):
    model.eval()
    model.n_substeps = cfg.n_substeps
    loss = Metric('mseloss')

    with torch.no_grad():
        # Start batch training
        infos = 'Val {} epoch {}/{}'.format('MPM only E', epoch, cfg.epochs)
        for init_pos, init_vel, init_ind, vel_field_gt in tqdm(val_loader, desc=infos):
            init_pos = init_pos.squeeze(0)
            init_vel = init_vel.squeeze(0)
            init_ind = init_ind.squeeze(0)
            vel_field_gt = vel_field_gt.squeeze(0).cuda()

            num = init_pos.size(0)
            C = torch.zeros((num, cfg.dim, cfg.dim))
            J = torch.ones((num))

            model.set_input([init_pos, init_vel, init_ind, C, J])
            vel_field_pred = model.forward()
            masked_pred = vel_field_pred[:, cfg.bound:cfg.n_grid[0] - cfg.bound,
                          cfg.bound:cfg.n_grid[1] - cfg.bound, :]
            masked_gt = vel_field_gt[:, cfg.bound:cfg.n_grid[0] - cfg.bound,
                        cfg.bound:cfg.n_grid[1] - cfg.bound, :]
            mseLoss = mse_loss(masked_pred, masked_gt)
            loss.update(mseLoss.detach().cpu(), 1)
    return loss.avg


if __name__ == '__main__':
    # Initialize random seeds
    seed_everything(cfg.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # define some parameters
    cfg.data_path = '../../data/exp_data/'
    cfg.E = 2000
    cfg.w_ext = 0.03
    cfg.n_substeps = 60
    cfg.st_epoch = 0
    cfg.epochs = 200
    cfg.lr_decay_freq = 50
    cfg.eval_freq = 1
    cfg.save_freq = 20
    cfg.lr = 0.0001
    best_loss = 1e+10

    root_folder = f'checkpoints_E{cfg.E}_w_ext{cfg.w_ext}_clap0.01'
    cfg.save_root = f'{root_folder}/exp1_lr{cfg.lr}_lambda0.9_r5'
    os.makedirs(cfg.save_root, exist_ok=True)
    log_file = Log(cfg.save_root)

    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in vars(cfg).items():
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    log_file.log(message)

    writer = SummaryWriter(cfg.save_root)
    model = MPM(cfg).cuda() # Initialize model with dataparallel (according to the # of GPUs available)

    total_params = count_parameters(model) # Count total parameters in the model
    print(f'Total trainable parameters: {total_params}')

    if os.path.exists(cfg.save_root + '/mpm_latest.pth'):
        state_data = torch.load(cfg.save_root + '/mpm_latest.pth')
        best_loss = state_data['best_loss']
        cfg.st_epoch = state_data['epoch']
        model.load_state_dict(state_data['state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** (epoch / cfg.lr_decay_freq))

    train_dataset = Dataset(cfg.data_path, phase='train_60')
    val_dataset = Dataset(cfg.data_path, phase='val_60')
    print('data length of train dataset: {}'.format(len(train_dataset)))
    print('data length of val dataset: {}'.format(len(val_dataset)))
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              num_workers=4,
                              pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=False)

    # Start the training
    log_file.log('Begin training')
    for epoch in range(cfg.st_epoch):
        scheduler.step()

    for epoch in range(cfg.st_epoch, cfg.epochs):
        # Start batch training
        start = time()
        curr_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = train(model, optimizer, scheduler, train_loader, cfg, epoch + 1)
        log_file.log(
            'epoch|epochs:{}|{}, current learning rate: {:0.8f}, train_loss:{:0.8f}, time: {:0.8f}'.format(
                epoch + 1, cfg.epochs,
                curr_lr,
                avg_train_loss.numpy(),
                time() - start))

        writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)

        torch.save({'state_dict': model.state_dict(),
                    'best_loss': best_loss, 'epoch': epoch + 1},
                   os.path.join(cfg.save_root, 'mpm_latest.pth'))

        if (epoch + 1) % cfg.save_freq == 0:
            torch.save({'state_dict': model.state_dict(),
                        'best_loss': best_loss, 'epoch': epoch + 1},
                       os.path.join(cfg.save_root, f'mpm_{epoch + 1}.pth'))
            log_file.log('save the model: epoch {}'.format(epoch + 1))

        log_file.log('E:{}'.format(model.E.view(-1).detach().cpu().numpy()), flag=False)

        if (epoch + 1) % cfg.eval_freq == 0 or epoch == 0:
            start = time()
            curr_lr = optimizer.param_groups[0]['lr']
            avg_val_loss = val(model, val_loader, cfg, epoch + 1)
            log_file.log(
                'epoch|epochs:{}|{}, current learning rate: {:0.8f},test_loss:{:0.8f}, time: {:0.8f}'.format(
                    epoch + 1, cfg.epochs,
                    curr_lr,
                    avg_val_loss.numpy(),
                    time() - start))
            writer.add_scalar('Loss/test', avg_val_loss, epoch + 1)

            if avg_val_loss > 0 and avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save({'state_dict': model.state_dict(),
                            'best_loss': best_loss, 'epoch': epoch + 1},
                           os.path.join(cfg.save_root, 'mpm_best.pth'))
                log_file.log('save the best model: epoch {}'.format(epoch + 1))
            log_file.log('\n')
