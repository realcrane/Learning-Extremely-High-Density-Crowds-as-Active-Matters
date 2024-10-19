import torch
import torch.nn as nn


class Parameter_Estimate(nn.Module):

    def __init__(self,
                 in_channels,
                 img_size,
                 hidden_dims=None):
        super(Parameter_Estimate, self).__init__()
        self.img_size = img_size

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 64, 32]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.Tanh())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1],
                      1,
                      kernel_size=3,
                      stride=1,
                      padding=1))

    def forward(self, input):
        x = input.unsqueeze(0).permute(0, 3, 1, 2)
        x = self.encoder(x)
        output = self.final_layer(x)
        return output.view(-1, 1)
