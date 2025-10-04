import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.path_finder import get_meg_fmri_paths
from data.dataset import FMRI_MEG_Dataset
from pix2pix import FCGenerator
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_dir = "/run/media/buura/EXTERNAL_USB/test_subjects/"
model_path = "./checkpoints/generator_epoch125.pth"
window_size = 64
batch_size = 32

pairs = get_meg_fmri_paths(base_dir)
fmri_paths = [pair[1] for pair in pairs]
meg_paths = [pair[0] for pair in pairs]
dataset = FMRI_MEG_Dataset(fmri_paths, meg_paths, window_size=window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

generator = FCGenerator(input_dim=328, output_size=(32, 32)).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

mse_criterion = nn.MSELoss()
total_mse = 0.0
total_ssim = 0.0
num_samples = 0

with torch.no_grad():
    for meg_slice, fmri_slice in dataloader:
        meg_slice = meg_slice.to(device)
        fmri_slice = fmri_slice.to(device).unsqueeze(1)

        fake_fmri = generator(meg_slice)

        mse = mse_criterion(fake_fmri, fmri_slice).item()
        total_mse += mse * meg_slice.size(0)

        fake_np = fake_fmri.cpu().numpy()
        real_np = fmri_slice.cpu().numpy()

        for i in range(fake_np.shape[0]):
            fake_img = np.squeeze(fake_np[i])
            real_img = np.squeeze(real_np[i])
            ssim_val = ssim(real_img, fake_img, data_range=real_img.max() - real_img.min())
            total_ssim += ssim_val
            num_samples += 1

# Average metrics
avg_mse = total_mse / num_samples
avg_ssim = total_ssim / num_samples

print(f"Average MSE: {avg_mse:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
