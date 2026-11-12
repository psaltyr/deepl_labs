from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from split import CLASSES, load_split

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 160


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class SplitDataset(Dataset):
    """Дані завантажуються лише за split.csv."""

    def __init__(self, data_root, part, transform, split_path="split.csv"):
        data_root = Path(data_root)
        folder = {"train": "seg_train", "validation": "seg_test", "test": "seg_test"}[part]
        self.items = [
            (data_root / folder / folder / label / file, CLASSES.index(label))
            for file, label, p in load_split(split_path) if p == part
        ]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def make_loaders(data_root, split_path="split.csv", with_test=False):
    """train: batch 32, shuffle, без власного generator. val/test: batch 64, без shuffle."""
    train = SplitDataset(data_root, "train", train_transform(), split_path)
    val = SplitDataset(data_root, "validation", eval_transform(), split_path)
    train_loader = DataLoader(train, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=0)
    if not with_test:
        return train_loader, val_loader
    test = SplitDataset(data_root, "test", eval_transform(), split_path)
    return train_loader, val_loader, DataLoader(test, batch_size=64, shuffle=False, num_workers=0)


def save_batch_figure(loader, path="batch_after_transforms.png"):
    x, y = next(iter(loader))
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, img, label in zip(axes.flat, x[:16], y[:16]):
        ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[label], fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    torch.manual_seed(1)
    train_loader, val_loader = make_loaders(args.data)
    print("train:", len(train_loader.dataset), "val:", len(val_loader.dataset))
    save_batch_figure(train_loader)