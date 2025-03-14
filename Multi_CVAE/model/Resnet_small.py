import torch
from torch import nn
import torch.nn.functional as F


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale_factor=2):
        super().__init__()
        if stride == 1:
            self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.conv1 = ResizeConv2d(out_planes, out_planes, kernel_size=3, scale_factor=scale_factor)
            self.bn1 = nn.BatchNorm2d(out_planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, out_planes, kernel_size=3, scale_factor=scale_factor),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, z_dim=128, nc=3):
        super().__init__()
        self.in_planes = 16
        self.z_dim = z_dim
        self.normalization = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(nc, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(*[BasicBlockEnc(in_planes=16, out_planes=32, stride=2)])
        self.layer2 = nn.Sequential(*[BasicBlockEnc(in_planes=32, out_planes=64, stride=2)])
        self.layer3 = nn.Sequential(*[BasicBlockEnc(in_planes=64, out_planes=128, stride=2)])
        self.drop = nn.Dropout(0.3)
        self.linear = nn.Linear(128, 2 * z_dim)

    def forward(self, x, c):
        c = F.interpolate(c.view(-1, 1, 6, 4), scale_factor=5, mode='nearest')
        x = self.normalization(x)
        x_c = torch.cat([x, c], 1)
        x = torch.relu(self.bn1(self.conv1(x_c)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        # x = self.drop(x)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class ResNet18Emb(nn.Module):

    def __init__(self, z_dim=64, nc=8):
        super().__init__()
        self.in_planes = 16
        self.z_dim = z_dim
        self.normalization = nn.BatchNorm2d(8)
        self.conv1 = nn.Conv2d(nc, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(*[BasicBlockEnc(in_planes=16, out_planes=32, stride=2)])
        self.layer2 = nn.Sequential(*[BasicBlockEnc(in_planes=32, out_planes=64, stride=2)])
        self.layer3 = nn.Sequential(*[BasicBlockEnc(in_planes=64, out_planes=128, stride=2)])
        self.linear = nn.Linear(128, z_dim)

    def forward(self, x):
        x = self.normalization(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet18Dec(nn.Module):

    def __init__(self, z_dim=10, nc=2):
        super().__init__()
        self.in_planes = z_dim
        self.linear = nn.Linear(z_dim + 24, z_dim)
        self.layer3 = nn.Sequential(*[BasicBlockDec(in_planes=z_dim, out_planes=64, stride=2, scale_factor=(2, 5 / 3))])
        self.layer2 = nn.Sequential(*[BasicBlockDec(in_planes=64, out_planes=32, stride=2, scale_factor=(15 / 8, 2))])
        self.layer1 = nn.Sequential(*[BasicBlockDec(in_planes=32, out_planes=16, stride=2, scale_factor=2)])
        self.conv1 = nn.Conv2d(16, nc, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), -1, 1, 1)
        x = F.interpolate(x, scale_factor=(4, 3))
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        return x


class CVAE(nn.Module):

    def __init__(self, z_dim, n_decoder):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.embedding = ResNet18Emb(z_dim=24)
        self.fc_alpha = nn.Linear(z_dim + 24, n_decoder)
        self.decoders = []
        module = []
        for i in range(n_decoder):
            module.append(ResNet18Dec(z_dim=z_dim))
        self.decoders = nn.ModuleList(module)

    def forward(self, x, cond):
        c = self.embedding(cond)
        mean, logvar = self.encoder(x, c)
        z = self.reparameterize(mean, logvar)
        z_c = torch.cat([z, c], dim=1)

        alphas = F.softmax(self.fc_alpha(z_c), dim=1)
        recon_x = 0
        for i, decoder in enumerate(self.decoders):
            recon_x += alphas[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) * decoder(z_c)
        return recon_x, mean, logvar

    def validate(self, x, cond):
        c = self.embedding(cond)
        mean, logvar = self.encoder(x, c)
        z_c = torch.cat([mean, c], dim=1)

        alphas = F.softmax(self.fc_alpha(z_c), dim=1)
        recon_x = 0
        for i, decoder in enumerate(self.decoders):
            recon_x += alphas[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) * decoder(z_c)
        return recon_x, mean, logvar

    def sample(self, cond):
        num_samples = cond.size(0)
        z = torch.randn(num_samples, self.z_dim).to(cond.device)

        c = self.embedding(cond)
        z_c = torch.cat([z, c], dim=1)
        alphas = F.softmax(self.fc_alpha(z_c), dim=1)
        recon_x = 0
        for i, decoder in enumerate(self.decoders):
            recon_x += alphas[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) * decoder(z_c)
        return recon_x

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean
