import os
import pandas as pd
import mne
import torch
from data.preprocess import load_fmri, load_meg, normalize_fmri, normalize_meg
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import Resize
mne.set_log_level("WARNING")

# Lots of bottlenecks here
class FMRI_MEG_Dataset(Dataset):
    def __init__(self, fmri_paths, meg_paths, window_size=64, preload=False):
        self.fmri_paths = fmri_paths
        self.meg_paths = meg_paths
        self.window_size = window_size
        self.preload = preload
        self.resize_transform = Resize((32, 32)) 
        if self.preload:
            self.fmri_data_list = []
            self.meg_data_list = []
            self.meg_raw_list = []
            for fmri_path, meg_path in zip(fmri_paths, meg_paths):
                fmri = normalize_fmri(load_fmri(fmri_path))
                meg, raw = load_meg(meg_path)
                meg = normalize_meg(meg)
                self.fmri_data_list.append(fmri)
                self.meg_data_list.append(meg)
                self.meg_raw_list.append(raw)

    def __getitem__(self, idx):
        fmri_path = self.fmri_paths[idx]
        meg_path = self.meg_paths[idx]

        if self.preload:
            fmri_data = self.fmri_data_list[idx]
            meg_data = self.meg_data_list[idx]
            raw_meg = self.meg_raw_list[idx]
        else:
            fmri_data = normalize_fmri(load_fmri(fmri_path))
            meg_data, raw_meg = load_meg(meg_path)
            meg_data = normalize_meg(meg_data)

        aligned = self.align_meg_to_fmri_stimuli(raw_meg, fmri_path)
        if aligned is None:
            raise ValueError(f"Could not align MEG and fMRI for {fmri_path}")
        _, event_time_meg = aligned[0]
        event_time_fmri = self.get_fmri_event_times(fmri_path)[0]

        fmri_slice = self.slice_fmri_from_event(fmri_data, event_time_fmri)
        meg_slice = self.slice_meg_from_event(meg_data, event_time_meg)
        meg_signal = meg_slice.mean(dim=0)  # (T,)
        meg_signal = (meg_signal - meg_signal.mean()) / meg_signal.std()

        # Correlate each Z slice
        correlations = []
        for z in range(fmri_slice.shape[2]):
            slice_data = fmri_slice[:, :, z, :]  # (X, Y, T)
            ts = slice_data.mean(dim=(0, 1))          # (T,)
            ts = (ts - ts.mean()) / ts.std()
            corr = torch.corrcoef(torch.stack([ts, meg_signal[:ts.shape[0]]]))[0, 1]
            correlations.append(corr.item())

        correlations = torch.tensor(correlations)
        best_z = correlations.argmax().item()
        best_fmri_slice = fmri_slice[:, :, best_z, 32]  # (X=106, Y=106)
        best_fmri_slice = best_fmri_slice.unsqueeze(0).unsqueeze(0)  # [1, 1, 106, 106]

        resized_fmri = torch.nn.functional.interpolate(
            best_fmri_slice, size=(32, 32), mode="bilinear", align_corners=False
        ).squeeze()  # [32, 32]

        # Get MEG at time 32
        meg_t = meg_slice[:, 32]  # [328]
        return meg_t, resized_fmri

    def __len__(self):
        return len(self.fmri_paths)

    def slice_meg_from_event(self, meg_data, event_time):
        start_idx = int(event_time)
        end_idx = start_idx + self.window_size
        meg_tensor = torch.tensor(meg_data).float()
        slice_ = meg_tensor[:, start_idx:end_idx]
        if slice_.shape[-1] < self.window_size:
            slice_ = F.pad(slice_, (0, self.window_size - slice_.shape[-1]))
        return slice_

    def align_meg_to_fmri_stimuli(self, raw, fmri_path):
        event_file = fmri_path.replace('_bold.nii.gz', '_events.tsv')
        if not os.path.exists(event_file):
            print(f"[Warning] Event file not found: {event_file}")
            return None

        fmri_df = pd.read_csv(event_file, sep='\t')
        stim_files = list(fmri_df['stim_file'])[:1]
        fmri_onsets = fmri_df['onset'].values[:1]

        events = mne.find_events(raw, stim_channel='STI101', verbose=False)
        meg_events = events[events[:, 2] <= 2]
        meg_onsets = meg_events[:, 0] / raw.info['sfreq']

        if len(fmri_onsets) != 1 or len(meg_onsets) < 1:
            print(f"⚠️  Mismatch: {len(fmri_onsets)} fMRI onsets vs {len(meg_onsets)} MEG events")
            return None

        return [(stim_files[0], meg_onsets[0])]

    def get_fmri_event_times(self, fmri_path):
        event_file = fmri_path.replace('_bold.nii.gz', '_events.tsv')
        df = pd.read_csv(event_file, sep='\t')
        return df['onset'].values[:1]

# UNUSED RN used for timing visualization before
    def plot_event_timings(self, event_times_meg, event_times_fmri):
        """Plot event timings for MEG and fMRI to compare"""
        plt.figure(figsize=(12, 4))
        plt.eventplot([event_times_meg, event_times_fmri], 
                      lineoffsets=[1, 0], colors=['blue', 'red'])
        plt.yticks([1, 0], ['MEG', 'fMRI'])
        plt.xlabel('Time (seconds)')
        plt.title('Event Timing Comparison: MEG vs fMRI')
        plt.grid(True)
        plt.show()
    
    def slice_fmri_from_event(self, fmri_data, event_time):
        """Slice 64 timepoints of fMRI starting from the event_time (no centering)"""
        timepoints = fmri_data.shape[-1]
        start_idx = int(event_time)
        end_idx = start_idx + self.window_size

        if end_idx > timepoints:
            # Pad if needed
            pad_width = end_idx - timepoints
            fmri_slice = fmri_data[..., start_idx:]
            fmri_slice = np.pad(fmri_slice, ((0, 0), (0, 0), (0, 0), (0, pad_width)), mode='constant')
        else:
            fmri_slice = fmri_data[..., start_idx:end_idx]

        return torch.tensor(fmri_slice).float()

