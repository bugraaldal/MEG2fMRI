import os

def get_meg_fmri_paths(base_dir):
    """
    Pairs MEG and fMRI paths by subject ID. 
    Only includes pairs where the fMRI file has a corresponding _events.tsv file.
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

        meg_files = sorted([f for f in os.listdir(meg_folder) if f.endswith("_meg.fif")])
        func_files = sorted([f for f in os.listdir(func_folder) if f.endswith("_bold.nii.gz")])

        for func_file, meg_file in zip(func_files, meg_files):
            fmri_event_file = func_file.replace("_bold.nii.gz", "_events.tsv")
            fmri_event_path = os.path.join(func_folder, fmri_event_file)

            if os.path.exists(fmri_event_path):
                fmri_path = os.path.join(func_folder, func_file)
                meg_path = os.path.join(meg_folder, meg_file)
                pairs.append((meg_path, fmri_path))
    return pairs
