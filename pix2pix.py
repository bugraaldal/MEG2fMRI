import torch
import torch.nn as nn

class FCGenerator(nn.Module):
    def __init__(self, input_dim=328, output_size=(32, 32)):
        super(FCGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_size[0] * output_size[1]),
            nn.Tanh()
        )
        self.output_size = output_size

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, self.output_size[0], self.output_size[1])
        return x


class HybridDiscriminator(nn.Module):
    def __init__(self, meg_dim=328, fmri_shape=(32, 32)):
        super(HybridDiscriminator, self).__init__()
        self.meg_project = nn.Linear(meg_dim, fmri_shape[0] * fmri_shape[1])
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, meg, fmri):
        meg_map = self.meg_project(meg)
        meg_map = meg_map.view(meg.size(0), 1, 32, 32)
        x = torch.cat([meg_map, fmri], dim=1)
        return self.conv(x)


class Pix2Pix(nn.Module):
    def __init__(self, generator, discriminator):
        super(Pix2Pix, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, meg):
        gen_output = self.generator(meg)
        disc_output = self.discriminator(meg, gen_output)
        return gen_output, disc_output