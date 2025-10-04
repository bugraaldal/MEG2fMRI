import mne
import nibabel as nib
import numpy as np
mne.set_log_level("WARNING") 

def load_meg(filepath, resample_rate=100):
    raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    raw.resample(resample_rate)
    data = raw.get_data()
    return data, raw

def load_fmri(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)
    return data

def normalize_meg(meg):
    mean = np.mean(meg, axis=1, keepdims=True)
    std = np.std(meg, axis=1, keepdims=True)
    return (meg - mean) / (std + 1e-8)

def normalize_fmri(fmri):
    fmri_min = np.min(fmri)
    fmri_max = np.max(fmri)
    return (fmri - fmri_min) / (fmri_max - fmri_min + 1e-8)