# data/path_finder.py

import os

def get_meg_fmri_paths(base_dir):
    """
    Finds and pairs MEG and fMRI paths by subject ID.
    Returns a list of (meg_path, fmri_path) tuples.
    """
    pairs = []

    for subject_dir in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        
        meg_folder = os.path.join(subject_path, "meg")
        func_folder = os.path.join(subject_path, "func")

        if not (os.path.exists(meg_folder) and os.path.exists(func_folder)):
            continue

        # Find MEG and fMRI files
        meg_files = [f for f in os.listdir(meg_folder) if f.endswith("_meg.fif")]
        func_files = [f for f in os.listdir(func_folder) if f.endswith("_bold.nii.gz")]
        n_meg_files, n_func_files = len(meg_files), len(func_files)
        if len(meg_files) == 0 or len(func_files) == 0:
            continue
        
        if n_meg_files > n_func_files:
            smaller_mod = n_func_files
        else:
            smaller_mod = n_meg_files


        for i in range(smaller_mod):
            meg_path = os.path.join(meg_folder, meg_files[i])
            fmri_path = os.path.join(func_folder, func_files[i])
            pairs.append((meg_path, fmri_path))
    print(len(pairs))
    return pairs
