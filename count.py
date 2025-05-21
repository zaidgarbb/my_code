import os
import nibabel as nib
import numpy as np
import imageio
from tqdm import tqdm

# 📁 مسارات الإدخال والإخراج
input_dir = "/Users/zaidraad/Desktop/BraTS2021_Extracted"
output_dir = "/Users/zaidraad/Desktop/BRATS_slices_classified"
os.makedirs(os.path.join(output_dir, "tumor"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "no_tumor"), exist_ok=True)

# 🎯 نوع الصورة المطلوبة (تقدر تغيرها لـ t1 أو t1ce أو t2)
modality = "_flair.nii"

# 🧮 عدادات
tumor_count = 0
no_tumor_count = 0

# 🧠 المرور على المرضى
for patient in tqdm(os.listdir(input_dir), desc="🧠 Processing patients"):
    patient_path = os.path.join(input_dir, patient)
    flair_path = os.path.join(patient_path, patient + modality)
    seg_path = os.path.join(patient_path, patient + "_seg.nii")

    if not os.path.exists(flair_path) or not os.path.exists(seg_path):
        continue

    # تحميل الصور والتقنيع
    flair_img = nib.load(flair_path).get_fdata()
    seg_img = nib.load(seg_path).get_fdata()

    # المرور على كل slice
    for i in range(flair_img.shape[2]):
        flair_slice = flair_img[:, :, i]
        seg_slice = seg_img[:, :, i]

        # معالجة التحذير: لو الصورة كلها نفس القيمة
        if flair_slice.max() != flair_slice.min():
            flair_slice_norm = np.uint8(255 * (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min()))
        else:
            flair_slice_norm = np.uint8(flair_slice)  # أو np.zeros_like(flair_slice, dtype=np.uint8)

        # التصنيف حسب وجود ورم في الـ seg
        if np.any(seg_slice > 0):
            category = "tumor"
            tumor_count += 1
        else:
            category = "no_tumor"
            no_tumor_count += 1

        # حفظ الصورة بصيغة PNG
        filename = f"{patient}_slice_{i:03d}.png"
        imageio.imwrite(os.path.join(output_dir, category, filename), flair_slice_norm)

# 🧾 التقرير النهائي
print("\n✅ Done!")
print(f"🧠 Tumor slices saved:     {tumor_count}")
print(f"🧼 No tumor slices saved:  {no_tumor_count}")
