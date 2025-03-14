from utils.utils import Log, seed_everything, MetricList, count_parameters, mkdir
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from dataset.cvae_dataset import CVAEDataset
import sys

sys.path.append('../../')
from config import config as cfg
from model.Resnet_small import CVAE
from time import time
from tqdm import tqdm
import torch
import os

def loss_function(recons, input, mu, logvar, kld_weight=0.5):
    recons_loss = mse_loss(recons, input, reduction='sum') / recons.size(0)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

    loss = recons_loss + kld_weight * kld_loss
    return loss, recons_loss, kld_loss


def train(model, optimizer, scheduler, train_loader, cfg, epoch):
    model.train()
    losses = MetricList(['loss', 'recons_loss', 'kld_loss'])

    # Start batch training
    infos = 'Train {} epoch {}/{}'.format('CVAE', epoch, cfg.epochs)
    for input, cond in tqdm(train_loader, desc=infos):
        input = input.view(-1, 2, 30, 20).cuda()
        cond = cond.view(-1, 8, 30, 20).cuda()

        recons, mu, logvar = model.forward(input, cond)
        loss, recons_loss, kld_loss = loss_function(recons, input, mu, logvar, kld_weight=cfg.kld_weight)
        losses.update([loss, recons_loss, kld_loss], input.size(0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()
    scheduler.step()
    return losses


def val(model, val_loader, cfg, epoch):
    model.eval()
    losses = MetricList(['loss', 'recons_loss', 'kld_loss'])

    # Start batch training
    infos = 'Val {} epoch {}/{}'.format('CVAE', epoch, cfg.epochs)
    with torch.no_grad():
        for input, cond in tqdm(val_loader, desc=infos):
            input = input.view(-1, 2, 30, 20).cuda()
            cond = cond.view(-1, 8, 30, 20).cuda()

            recons, mu, logvar = model.validate(input, cond)
            loss, recons_loss, kld_loss = loss_function(recons, input, mu, logvar, kld_weight=cfg.kld_weight)
            losses.update([loss, recons_loss, kld_loss], input.size(0))
    return losses


if __name__ == '__main__':

    # Initialize random seeds
    seed_everything(cfg.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # define some parameters
    cfg.data_path = 'data/'
    cfg.st_epoch = 0
    cfg.epochs = 2000
    cfg.batch_size = 4
    cfg.lr_decay_freq = 20
    cfg.eval_freq = 5
    cfg.save_freq = 100
    cfg.n_decoder = 4
    cfg.lr = 0.0001
    cfg.kld_weight = 0.1
    cfg.lr_lambda = 0.9
    best_loss = 1e+10

    # creat folder for save
    root_folder = f'checkpoints{cfg.n_decoder}/resnet_sum_bs{cfg.batch_size}_latent64_small'
    cfg.save_folder = f'{root_folder}/exp1_kld_weight{cfg.kld_weight}'
    mkdir(root_folder)

    log_file = Log(cfg.save_folder)
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in vars(cfg).items():
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    log_file.log(message)

    writer = SummaryWriter(cfg.save_folder)

    # Initialize model with dataparallel (according to the # of GPUs available)
    model = CVAE(z_dim=64, n_decoder=cfg.n_decoder).cuda()
    model_path = os.path.join(cfg.save_folder, 'mpm_latest.pth')
    if os.path.exists(model_path):
        state_data = torch.load(model_path)
        best_loss = state_data['best_loss']
        cfg.st_epoch = state_data['epoch']
        model.load_state_dict(state_data['state_dict'])
        print('loading the model:{}'.format(model_path))
        print('the best loss:{}'.format(best_loss))
    else:
        print('--->no pretrained model to load')

    print(f'Total trainable parameters: {count_parameters(model)}')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: cfg.lr_lambda ** (epoch / cfg.lr_decay_freq))

    train_dataset = CVAEDataset(cfg.data_path, cfg, phase='train')
    val_dataset = CVAEDataset(cfg.data_path, cfg, phase='val')
    print('data length of train dataset: {}'.format(len(train_dataset)))
    print('data length of val dataset: {}'.format(len(val_dataset)))

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              num_workers=4,
                              pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True,
                            shuffle=True)

    # Start the training
    log_file.log('Begin training')
    for epoch in range(cfg.st_epoch):
        scheduler.step()

    for epoch in range(cfg.st_epoch, cfg.epochs):

        # Start batch training
        start = time()
        train_losses = train(model, optimizer, scheduler, train_loader, cfg, epoch + 1)
        log_file.log(
            'epoch|epochs:{}|{}, current learning rate: {:0.8f}, kld_weight: {:0.8f}, train_loss:{:0.8f}, recons_loss:{:0.8f}, kld_loss:{:0.8f}, time: {:0.8f}'.format(
                epoch + 1, cfg.epochs,
                optimizer.param_groups[0]['lr'],
                cfg.kld_weight,
                train_losses.avg['loss'],
                train_losses.avg['recons_loss'],
                train_losses.avg['kld_loss'],
                time() - start))

        writer.add_scalar('train/loss', train_losses.avg['loss'], epoch + 1)
        writer.add_scalar('train/recons_loss', train_losses.avg['recons_loss'], epoch + 1)
        writer.add_scalar('train/kld_loss', train_losses.avg['kld_loss'], epoch + 1)

        torch.save({'state_dict': model.state_dict(),
                    'best_loss': best_loss, 'epoch': epoch + 1},
                   os.path.join(cfg.save_folder, 'mpm_latest.pth'))

        if (epoch + 1) % cfg.eval_freq == 0:
            start = time()
            val_losses = val(model, val_loader, cfg, epoch + 1)
            log_file.log(
                'epoch|epochs:{}|{}, current learning rate: {:0.8f}, val_loss:{:0.8f}, recons_loss:{:0.8f}, kld_loss:{:0.8f}, time: {:0.8f}'.format(
                    epoch + 1, cfg.epochs,
                    optimizer.param_groups[0]['lr'],
                    val_losses.avg['loss'],
                    val_losses.avg['recons_loss'],
                    val_losses.avg['kld_loss'],
                    time() - start))

            writer.add_scalar('val/loss', val_losses.avg['loss'], epoch + 1)
            writer.add_scalar('val/recons_loss', val_losses.avg['recons_loss'], epoch + 1)
            writer.add_scalar('val/kld_loss', val_losses.avg['kld_loss'], epoch + 1)

            if best_loss > val_losses.avg['recons_loss']:
                best_loss = val_losses.avg['recons_loss']
                torch.save({'state_dict': model.state_dict(),
                            'best_loss': best_loss, 'epoch': epoch + 1},
                           os.path.join(cfg.save_folder, f'mpm_best.pth'))
                log_file.log('save the best model: epoch {}'.format(epoch + 1))
            log_file.log('\n')

        if (epoch + 1) % cfg.save_freq == 0:
            torch.save({'state_dict': model.state_dict(),
                        'best_loss': best_loss, 'epoch': epoch + 1},
                       os.path.join(cfg.save_folder, f'mpm_{epoch + 1}.pth'))
            log_file.log('save the model: epoch {}'.format(epoch + 1))
            log_file.log('\n')
