import os
import nibabel as nib
import numpy as np
import cv2
from glob import glob
import random

def extract_slices(brats_dir, output_dir, modalities=('t1', 't1ce', 't2', 'flair'), labeled_ratio=1.0):
    """
    Extracts 2D slices from 3D BraTS volumes, with options for modality selection and labeled data ratio.
    """
    image_dirs = sorted(glob(os.path.join(brats_dir, "*")))
    os.makedirs(os.path.join(output_dir, 'labeled/images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labeled/masks'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'unlabeled/images'), exist_ok=True)

    labeled_count = 0
    total_count = 0

    for case in image_dirs:
        case_id = os.path.basename(case)
        print(f"Processing case: {case_id}")
        try:
            # Load modalities
            data = {}
            if 't1' in modalities:
                data['t1'] = nib.load(glob(os.path.join(case, "*t1.nii"))[0]).get_fdata()
            if 't1ce' in modalities:
                data['t1ce'] = nib.load(glob(os.path.join(case, "*t1ce.nii"))[0]).get_fdata()
            if 't2' in modalities:
                data['t2'] = nib.load(glob(os.path.join(case, "*t2.nii"))[0]).get_fdata()
            if 'flair' in modalities:
                data['flair'] = nib.load(glob(os.path.join(case, "*flair.nii"))[0]).get_fdata()
            mask_img = nib.load(glob(os.path.join(case, "*seg.nii"))[0])
            mask = mask_img.get_fdata()

        except Exception as e:
            print(f"❌ Failed to load data for {case_id}: {e}")
            continue

        # Determine slice dimension
        slice_dim = np.argmin(mask_img.shape)
        slice_count = 0
        tumor_slices = 0
        slice_indices = list(range(mask_img.shape[slice_dim]))
        if labeled_ratio < 1.0:
            num_labeled = int(len(slice_indices) * labeled_ratio)
            labeled_indices = random.sample(slice_indices, num_labeled)
        else:
            labeled_indices = slice_indices

        for i in slice_indices:
            # Extract slice
            if slice_dim == 0:
                image_slices = [data[mod][i, :, :] for mod in modalities]
                mask_slice = mask[i, :, :]
            elif slice_dim == 1:
                image_slices = [data[mod][:, i, :] for mod in modalities]
                mask_slice = mask[:, i, :]
            else:
                image_slices = [data[mod][:, :, i] for mod in modalities]
                mask_slice = mask[:, :, i]
            image = np.stack(image_slices, axis=-1)
            image_resized = cv2.resize(image, (256, 256))
            mask_resized = cv2.resize(mask_slice, (256, 256), interpolation=cv2.INTER_NEAREST)
            has_tumor = np.max(mask_resized) > 0
            patient_id = os.path.basename(case)
            filename = f"{patient_id}_slice_{i:03d}.png"
            if i in labeled_indices and has_tumor:
                cv2.imwrite(os.path.join(output_dir, 'labeled/images', filename), image_resized)
                cv2.imwrite(os.path.join(output_dir, 'labeled/masks', filename), mask_resized)  # Save original mask values
                labeled_count += 1
                tumor_slices += 1
            else:
                cv2.imwrite(os.path.join(output_dir, 'unlabeled/images', filename), image_resized)
            slice_count += 1
            total_count += 1
        print(f"✅ {case_id}: {slice_count} slices, {tumor_slices} with tumor")
    print("\n📊 Final Summary")
    print(f"Total slices: {total_count}, Labeled (with tumor): {labeled_count}")

# Example usage:
extract_slices(
    "/Users/zaidraad/Desktop/BraTS2021_Extracted",
    "BraTS_FixMatch",
    modalities=('t1', 't1ce', 't2', 'flair'),  # Specify modalities
    labeled_ratio=0.1  # Example labeled ratio
)
