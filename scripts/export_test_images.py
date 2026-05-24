"""
scripts/export_test_images.py
Exports ChestMNIST test images organized by hospital profile for dashboard simulation.
Hospital A: clean reference
Hospital B: brightness degradation (covariate shift / incremental)
Hospital C: resolution downscale (covariate shift / sudden — applied after t=30 images)
Hospital D: rare-class oversampled (label shift / gradual)

Usage: python scripts/export_test_images.py
"""

import os, sys
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LABEL_NAMES = [
    'Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule',
    'Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema',
    'Fibrosis','Pleural_Thickening','Hernia'
]

# Classes considered "rare" for Hospital D (label shift simulation)
RARE_CLASSES = {'Hernia', 'Emphysema', 'Pneumonia', 'Fibrosis'}

HOSPITALS = {
    'hospital_A': 'Reference — no perturbation',
    'hospital_B': 'Covariate shift — brightness degradation (incremental)',
    'hospital_C': 'Covariate shift — resolution downscale (sudden after image 30)',
    'hospital_D': 'Label shift — rare class oversampling (pediatric profile)',
}

for h in HOSPITALS:
    os.makedirs(os.path.join(ROOT, 'test_images', h), exist_ok=True)

print("Loading ChestMNIST test set...")
from medmnist import ChestMNIST
test_ds = ChestMNIST(split="test", download=True)
print(f"  {len(test_ds)} test samples available\n")


def save(img: Image.Image, path: str):
    img.resize((224, 224), Image.LANCZOS).save(path)


def perturb_B(img: Image.Image, index: int) -> Image.Image:
    """Incremental brightness drop: factor goes from 1.0 down to 0.4 over 60 images."""
    factor = max(0.4, 1.0 - (index / 60) * 0.6)
    return ImageEnhance.Brightness(img).enhance(factor)


def perturb_C(img: Image.Image, index: int) -> Image.Image:
    """Sudden resolution degradation after image 30: downscale to 14x14 then back."""
    if index < 30:
        return img
    small = img.resize((14, 14), Image.BILINEAR)
    return small.resize((224, 224), Image.NEAREST)


def perturb_D(img: Image.Image) -> Image.Image:
    """Mild contrast shift to simulate different patient demographics."""
    return ImageEnhance.Contrast(img).enhance(0.75)


# ── Collect images ─────────────────────────────────────────────────────────────

def label_vec_to_set(label_vec):
    return {LABEL_NAMES[j] for j, v in enumerate(label_vec) if v == 1}

# We collect two pools:
# - general_pool: up to 60 diverse images for A/B/C
# - rare_pool: up to 60 images where at least one RARE_CLASS is positive (for D)

general_pool = []   # list of PIL images + metadata
rare_pool    = []

TARGET_GENERAL = 60
TARGET_RARE    = 60

print("Collecting images from test set...")
for i in range(len(test_ds)):
    img, label = test_ds[i]
    label_vec  = label.flatten().tolist()
    positives  = label_vec_to_set(label_vec)
    label_str  = "_".join(sorted(positives)) if positives else "NoFinding"

    entry = {
        'img':       img,  # PIL image (28x28 grayscale)
        'idx':       i,
        'positives': positives,
        'label_str': label_str,
    }

    if len(general_pool) < TARGET_GENERAL:
        general_pool.append(entry)

    if len(rare_pool) < TARGET_RARE and positives & RARE_CLASSES:
        rare_pool.append(entry)

    if len(general_pool) >= TARGET_GENERAL and len(rare_pool) >= TARGET_RARE:
        break

print(f"  General pool: {len(general_pool)} images")
print(f"  Rare pool:    {len(rare_pool)} images\n")


# ── Hospital A — clean reference ───────────────────────────────────────────────
print("Exporting Hospital A (reference)...")
for rank, entry in enumerate(general_pool):
    fname = f"{rank:03d}_{entry['idx']:05d}_{entry['label_str']}.png"
    save(entry['img'], os.path.join(ROOT, 'test_images', 'hospital_A', fname))
print(f"  {len(general_pool)} images exported.\n")


# ── Hospital B — incremental brightness degradation ────────────────────────────
print("Exporting Hospital B (brightness degradation)...")
for rank, entry in enumerate(general_pool):
    img_p = perturb_B(entry['img'], rank)
    fname = f"{rank:03d}_{entry['idx']:05d}_{entry['label_str']}.png"
    save(img_p, os.path.join(ROOT, 'test_images', 'hospital_B', fname))
print(f"  {len(general_pool)} images exported.\n")


# ── Hospital C — sudden resolution drop after image 30 ─────────────────────────
print("Exporting Hospital C (sudden resolution drop at image 30)...")
for rank, entry in enumerate(general_pool):
    img_p = perturb_C(entry['img'], rank)
    fname = f"{rank:03d}_{entry['idx']:05d}_{entry['label_str']}.png"
    save(img_p, os.path.join(ROOT, 'test_images', 'hospital_C', fname))
print(f"  {len(general_pool)} images exported (drift injected at index 30).\n")


# ── Hospital D — rare class oversampling + contrast shift ──────────────────────
print("Exporting Hospital D (rare class / label shift)...")
# Interleave rare_pool images 2:1 with general to simulate prevalence shift
d_pool = []
ri, gi = 0, 0
while len(d_pool) < TARGET_GENERAL:
    # 2 rare, 1 general, repeat
    for _ in range(2):
        if ri < len(rare_pool):
            d_pool.append(rare_pool[ri]); ri += 1
    if gi < len(general_pool):
        d_pool.append(general_pool[gi]); gi += 1

d_pool = d_pool[:TARGET_GENERAL]

for rank, entry in enumerate(d_pool):
    img_p = perturb_D(entry['img'])
    fname = f"{rank:03d}_{entry['idx']:05d}_{entry['label_str']}.png"
    save(img_p, os.path.join(ROOT, 'test_images', 'hospital_D', fname))
print(f"  {len(d_pool)} images exported.\n")


# ── Summary ────────────────────────────────────────────────────────────────────
print("=" * 55)
print("Export complete. Structure:")
for h, desc in HOSPITALS.items():
    d = os.path.join(ROOT, 'test_images', h)
    n = len(os.listdir(d))
    print(f"  {h}/  ({n} images) — {desc}")
print("\nRun dashboard/app.py to start the simulation.")