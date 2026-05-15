import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import ChestMNIST, INFO

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Label info ─────────────────────────────────────────────────────────────────
INFO_DATA = INFO["chestmnist"]
NUM_CLASSES = len(INFO_DATA["label"])
LABEL_NAMES = list(INFO_DATA["label"].values())
# 14 pathology labels (multi-label binary classification)

# ── Transforms ─────────────────────────────────────────────────────────────────
def get_transforms(split: str, image_size: int = 224):
    """
    Returns transforms for the given split.
    ChestMNIST images are 28x28 grayscale — we resize and convert to 3-channel
    for compatibility with pretrained backbones (MobileNet, ResNet, etc.)
    """
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),  # 1 → 3 channels
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])

# ── DataLoader factory ──────────────────────────────────────────────────────────
def get_dataloaders(
    download: bool = True,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    root: str = "data",
    seed: int = SEED,
):
    """
    Builds and returns train, val, and test DataLoaders for ChestMNIST.

    Args:
        download    : download the dataset if not already present
        image_size  : resize target (224 for pretrained backbones)
        batch_size  : samples per batch
        num_workers : parallel data loading workers (use 0 on Windows)
        root        : directory where medmnist data is stored
        seed        : random seed for reproducibility

    Returns:
        dict with keys 'train', 'val', 'test' → DataLoader
                       'num_classes'           → int (14)
                       'label_names'           → list of str
    """
    set_seed(seed)

    train_ds = ChestMNIST(
        split="train",
        transform=get_transforms("train", image_size),
        download=download,
        root=root,
    )
    val_ds = ChestMNIST(
        split="val",
        transform=get_transforms("val", image_size),
        download=download,
        root=root,
    )
    test_ds = ChestMNIST(
        split="test",
        transform=get_transforms("test", image_size),
        download=download,
        root=root,
    )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=lambda wid: np.random.seed(seed + wid),
        )

    return {
        "train":       make_loader(train_ds, shuffle=True),
        "val":         make_loader(val_ds,   shuffle=False),
        "test":        make_loader(test_ds,  shuffle=False),
        "num_classes": NUM_CLASSES,
        "label_names": LABEL_NAMES,
    }


# ── Quick sanity check ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from collections import Counter

    loaders = get_dataloaders(batch_size=8, num_workers=0)

    # ── Basic shapes ───────────────────────────────────────────────────────────
    images, labels = next(iter(loaders["train"]))
    print(f"[train] batch shape  : {images.shape}")       # (8, 3, 224, 224)
    print(f"[train] labels shape : {labels.shape}")       # (8, 14)
    print(f"num classes          : {loaders['num_classes']}")
    print(f"label names          : {loaders['label_names']}")

    # ── Split sizes ────────────────────────────────────────────────────────────
    print(f"\n[split sizes]")
    print(f"  train : {len(loaders['train'].dataset)}")   # 78468
    print(f"  val   : {len(loaders['val'].dataset)}")     # 11219
    print(f"  test  : {len(loaders['test'].dataset)}")    # 22433

    # ── Pixel value range after normalization ──────────────────────────────────
    print(f"\n[pixel stats]")
    print(f"  min : {images.min():.3f}")
    print(f"  max : {images.max():.3f}")
    print(f"  mean: {images.mean():.3f}")

    # ── Label distribution (class imbalance check) ─────────────────────────────
    print(f"\n[label distribution on first batch]")
    for i, name in enumerate(loaders["label_names"]):
        count = int(labels[:, i].sum().item())
        print(f"  {name:<20} : {count}/8 positive")

    # ── Show sample images ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        # un-normalize for display: x = x*0.5 + 0.5
        img = images[i].permute(1, 2, 0) * 0.5 + 0.5
        img = img.numpy().clip(0, 1)
        ax.imshow(img[:, :, 0], cmap="gray")  # show 1 channel (they're identical)
        active = [loaders["label_names"][j] for j in range(14) if labels[i, j] == 1]
        ax.set_title("\n".join(active) if active else "No Finding", fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("figures/sample_batch.png", dpi=150)
    plt.show()
    print("\nSample batch saved to figures/sample_batch.png")