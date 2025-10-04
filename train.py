import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.path_finder import get_meg_fmri_paths
from data.dataset import FMRI_MEG_Dataset
from pix2pix import FCGenerator, HybridDiscriminator, Pix2Pix
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR


def save_comparison(real, generated, epoch, batch):
    batch_size = real.size(0)
    real = real[:batch_size].cpu().detach()
    generated = generated[:batch_size].cpu().detach()
    comparison = torch.cat([real, generated], dim=0)
    grid = vutils.make_grid(comparison, nrow=batch_size, normalize=True, scale_each=True)

    npimg = grid.numpy()
    plt.imsave(f"{output_dir}/compare_epoch{epoch}_batch{batch}.png", 
               np.transpose(npimg, (1, 2, 0)))


# Load data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = "/run/media/buura/EXTERNAL_USB/ds004078-download/"
output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("./checkpoints", exist_ok=True)
pairs = get_meg_fmri_paths(base_dir)
fmri_paths = [pair[1] for pair in pairs]
meg_paths = [pair[0] for pair in pairs]
dataset = FMRI_MEG_Dataset(fmri_paths, meg_paths, window_size=64)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Model
generator = FCGenerator(input_dim=328, output_size=(32, 32)).to(device)
discriminator = HybridDiscriminator(meg_dim=328, fmri_shape=(32, 32)).to(device)
model = Pix2Pix(generator, discriminator)
criterion = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

scheduler_g = StepLR(optimizer_g, step_size=20, gamma=0.5)
scheduler_d = StepLR(optimizer_d, step_size=20, gamma=0.5)

# Training
num_epochs = 125
for epoch in range(num_epochs):
    print('Epoch', epoch)
    for i, (meg_slice, fmri_slice) in enumerate(dataloader):
        meg_slice = meg_slice.to(device)
        fmri_slice = fmri_slice.to(device).unsqueeze(1)  # B, 1, 32, 32

        fake_fmri = generator(meg_slice).detach()
        real_pred = discriminator(meg_slice, fmri_slice)
        fake_pred = discriminator(meg_slice, fake_fmri)

        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)

        d_loss_real = bce_loss(real_pred, real_labels)
        d_loss_fake = bce_loss(fake_pred, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) * 0.5

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # GENERATOR TRAIN
        fake_fmri = generator(meg_slice)
        fake_pred = discriminator(meg_slice, fake_fmri)
        g_adv_loss = bce_loss(fake_pred, real_labels)
        g_recon_loss = criterion(fake_fmri, fmri_slice)  # Match real fMRI
        g_loss = g_adv_loss + g_recon_loss

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

        if i % 10 == 0:
            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
        if epoch % 50 == 0 and i == 0:
            save_comparison(fmri_slice, fake_fmri, epoch, i)
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), f"checkpoints/generator_epoch{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch{epoch+1}.pth")

    scheduler_g.step()
    scheduler_d.step()

print("Training completed.")