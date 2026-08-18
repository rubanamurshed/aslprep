"""
Build CBF Timeseries: Extract Parcel-Level CBF Timeseries per Subject

For each subject/session, loads the 4D CBF timeseries NIfTI, an atlas
parcellation, and a brain mask, then averages CBF within each atlas parcel
at every timepoint to produce a (time x parcels) array. This timeseries is
the input to run_cbf_graph_pipeline.py, which correlates parcels to build
the graph used in downstream network analyses.

Atlas: 4S256 (256-parcel)
"""

import os
import numpy as np
import nibabel as nib

ROOT = "/data/raulab/asl2/output"

# Subject codes only, no clinical/demographic data
SUBJECTS = [
    "sub-4028", "sub-4042", "sub-4045", "sub-4046", "sub-4049",
    "sub-4061", "sub-4062", "sub-4063", "sub-4067",
    "sub-5002", "sub-5008", "sub-5009", "sub-5012", "sub-5016",
]

SESSIONS = ["ses-01"]
ATLAS = "4S256"

# Adjust these names if your files differ
CBF_4D_NAME = "{sub}_{ses}_space-MNI152NLin2009aSym_desc-timeseries_cbf.nii.gz"

# NOTE: this currently points to ONE specific subject's (sub-5012) warped
# atlas file, applied to every subject in the loop below. Confirm this is
# intentional (e.g. a shared template-space atlas) rather than a leftover
# path that should instead be subject-specific.
ATLAS_NAME = "/data/raulab/asl2/aslwd/aslprep_25_1_wf/sub_5012_wf/asl_preproc_ses_01_wf/parcellate_cbf_wf/warp_atlases_to_asl_space/mapflow/_warp_atlases_to_asl_space2/tpl-MNI152NLin6Asym_atlas-4S256Parcels_res-01_dseg_trans.nii.gz"

MASK_NAME = "{sub}_{ses}_space-MNI152NLin2009aSym_desc-brain_mask.nii.gz"


def extract_timeseries(cbf_nii, atlas_nii, mask_nii):
    """
    cbf_nii: 4D NIfTI (x,y,z,time)
    atlas_nii: 3D NIfTI with integer parcel labels 1..N
    mask_nii: 3D NIfTI brain mask (0/1)
    returns: 2D array (time x parcels), mean CBF per parcel at each timepoint
    """
    cbf = cbf_nii.get_fdata()     # x,y,z,t
    atlas = atlas_nii.get_fdata() # x,y,z
    mask = mask_nii.get_fdata()   # x,y,z

    # apply mask
    mask_bool = mask > 0
    labels = np.unique(atlas[mask_bool])
    labels = labels[labels > 0]   # ignore 0/background
    n_parcels = int(labels.max())

    t = cbf.shape[3]
    ts = np.zeros((t, n_parcels), dtype=np.float32)

    for pid in range(1, n_parcels + 1):
        vox = (atlas == pid) & mask_bool
        if not np.any(vox):
            continue
        # mean across voxels in this parcel for each timepoint
        parcel_ts = cbf[vox, :]
        ts[:, pid - 1] = parcel_ts.mean(axis=0)

    return ts


def run_all_subjects():
    """Loop over every subject/session, extract their parcel-level CBF
    timeseries, and save it as a .npy file (skipping subjects already done)."""
    for sub in SUBJECTS:
        for ses in SESSIONS:
            perf_dir = os.path.join(ROOT, sub, ses, "perf")
            if not os.path.isdir(perf_dir):
                print(f"[SKIP] {sub} {ses}: no perf dir")
                continue

            out_npy = os.path.join(
                perf_dir, f"{sub}_{ses}_atlas-{ATLAS}_cbfTimeseries.npy"
            )
            if os.path.exists(out_npy):
                print(f"[INFO] {sub} {ses}: timeseries already exists, skipping")
                continue

            cbf_path = os.path.join(
                perf_dir, CBF_4D_NAME.format(sub=sub, ses=ses)
            )
            # ATLAS_NAME is an absolute path, so os.path.join here simply
            # returns ATLAS_NAME unchanged (perf_dir is effectively ignored)
            atlas_path = os.path.join(perf_dir, ATLAS_NAME)
            mask_path = os.path.join(
                perf_dir, MASK_NAME.format(sub=sub, ses=ses)
            )

            if not os.path.exists(cbf_path):
                print(f"[WARN] {sub} {ses}: missing 4D CBF {cbf_path}")
                continue
            if not os.path.exists(atlas_path):
                print(f"[WARN] {sub} {ses}: missing atlas {atlas_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"[WARN] {sub} {ses}: missing mask {mask_path}")
                continue

            print(f"[INFO] Extracting timeseries for {sub} {ses}")
            cbf_nii = nib.load(cbf_path)
            atlas_nii = nib.load(atlas_path)
            mask_nii = nib.load(mask_path)

            ts = extract_timeseries(cbf_nii, atlas_nii, mask_nii)
            np.save(out_npy, ts)
            print(f"   -> saved {out_npy}")


if __name__ == "__main__":
    run_all_subjects()
