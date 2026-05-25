"""
scripts/export_test_images.py — clean export, drift applied live in dashboard
"""
import os, sys
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LABEL_NAMES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
    'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
    'Fibrosis','Pleural_Thickening','Hernia'
]
RARE_CLASSES = {'Hernia', 'Emphysema', 'Pneumonia', 'Fibrosis'}

HOSPITALS = ['hospital_A', 'hospital_B', 'hospital_C', 'hospital_D']
for h in HOSPITALS:
    os.makedirs(os.path.join(ROOT, 'test_images', h), exist_ok=True)

print("Loading ChestMNIST test set...")
from medmnist import ChestMNIST
test_ds = ChestMNIST(split="test", download=True)

def label_vec_to_set(label_vec):
    return {LABEL_NAMES[j] for j, v in enumerate(label_vec) if v == 1}

general_pool = []
rare_pool    = []
TARGET = 120   # more images → longer simulation before cycling

print("Collecting images...")
for i in range(len(test_ds)):
    img, label = test_ds[i]
    label_vec  = label.flatten().tolist()
    positives  = label_vec_to_set(label_vec)
    label_str  = "_".join(sorted(positives)) if positives else "NoFinding"
    entry = {'img': img, 'idx': i, 'positives': positives, 'label_str': label_str}

    if len(general_pool) < TARGET:
        general_pool.append(entry)
    if len(rare_pool) < TARGET and positives & RARE_CLASSES:
        rare_pool.append(entry)
    if len(general_pool) >= TARGET and len(rare_pool) >= TARGET:
        break

def save_clean(img, path):
    img.resize((224, 224), Image.LANCZOS).save(path)

# All hospitals get the SAME clean images — perturbation happens live in dashboard
print("Exporting clean images to all hospitals...")
for h in HOSPITALS:
    pool = general_pool
    if h == 'hospital_D':
        # Interleave rare 2:1 for label shift — only the content changes, not appearance
        pool = []
        ri, gi = 0, 0
        while len(pool) < TARGET:
            for _ in range(2):
                if ri < len(rare_pool): pool.append(rare_pool[ri]); ri += 1
            if gi < len(general_pool): pool.append(general_pool[gi]); gi += 1
        pool = pool[:TARGET]

    for rank, entry in enumerate(pool):
        fname = f"{rank:03d}_{entry['idx']:05d}_{entry['label_str']}.png"
        save_clean(entry['img'], os.path.join(ROOT, 'test_images', h, fname))
    print(f"  {h}: {len(pool)} images")

print("\nDone. Drift is now applied live in the dashboard.")