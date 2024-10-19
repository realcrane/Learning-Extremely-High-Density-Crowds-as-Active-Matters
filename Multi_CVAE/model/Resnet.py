import torch
from torch import nn, optim
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

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, planes, stride=1, scale_factor=2):
        super().__init__()
        if stride == 1:
            self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv1 = ResizeConv2d(planes, planes, kernel_size=3, scale_factor=scale_factor)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=scale_factor),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=256, nc=3):
        super().__init__()
        self.in_planes = 32
        self.z_dim = z_dim
        self.normalizaton = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(nc, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(BasicBlockEnc, 32, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 64, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 128, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.linear = nn.Linear(256, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x[:, 0:2, :, :]
        x2 = x[:, 2:, :, :]
        x1 = self.normalizaton(x1)
        x = torch.cat([x1, x2], 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class ResNet18Emb(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=256, nc=8):
        super().__init__()
        self.in_planes = 32
        self.normalization1 = nn.BatchNorm2d(2)
        self.normalization2 = nn.BatchNorm2d(2)
        self.normalization3 = nn.BatchNorm2d(2)
        self.normalization4 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(nc, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(BasicBlockEnc, 32, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 64, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 128, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.linear = nn.Linear(256, z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.normalization1(x[:, 0:2, :, :])
        x2 = self.normalization2(x[:, 2:4, :, :])
        x3 = self.normalization3(x[:, 4:6, :, :])
        x4 = self.normalization4(x[:, 6:8, :, :])
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.linear(x))
        return x


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=2):
        super().__init__()
        self.in_planes = z_dim
        self.linear = nn.Linear(z_dim + 600, z_dim)
        self.layer4 = self._make_layer(BasicBlockDec, 128, num_Blocks[3], stride=2, scale_factor=(2, 5 / 3))
        self.layer3 = self._make_layer(BasicBlockDec, 64, num_Blocks[2], stride=2, scale_factor=(15 / 8, 2))
        self.layer2 = self._make_layer(BasicBlockDec, 32, num_Blocks[1], stride=2, scale_factor=2)
        self.layer1 = self._make_layer(BasicBlockDec, 32, num_Blocks[0], stride=1)
        self.conv1 = nn.Conv2d(32, nc, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride, scale_factor=2):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, planes, stride, scale_factor)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), -1, 1, 1)
        x = F.interpolate(x, scale_factor=(4, 3))
        x = self.layer4(x)
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
        self.embedding = ResNet18Emb(z_dim=600)
        self.fc_alpha = nn.Linear(z_dim + 600, n_decoder)
        self.decoders = []
        module = []
        for i in range(n_decoder):
            module.append(ResNet18Dec(z_dim=z_dim))
        self.decoders = nn.ModuleList(module)

    def forward(self, x, cond):
        c = self.embedding(cond)
        x_c = torch.cat([x, c.view(-1, 1, 30, 20)], dim=1)
        mean, logvar = self.encoder(x_c)
        z = self.reparameterize(mean, logvar)
        z_c = torch.cat([z, c], dim=1)

        alphas = F.softmax(self.fc_alpha(z_c), dim=1)
        recon_x = 0
        for i, decoder in enumerate(self.decoders):
            recon_x += alphas[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1) * decoder(z_c)
        return recon_x, mean, logvar

    def validate(self, x, cond):
        c = self.embedding(cond)
        x_c = torch.cat([x, c.view(-1, 1, 30, 20)], dim=1)
        mean, logvar = self.encoder(x_c)
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
