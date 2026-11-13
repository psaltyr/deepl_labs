import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from data import make_loaders
from model import build_model, get_device
from train import git_info, predict_logits

OUT = []


def out(s):
    print(s)
    OUT.append(s)


ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
args = ap.parse_args()

commit, dirty = git_info()
out(f"Додатковий дослід поза протоколом: frozen, seed 1, model.train() замість set_train_mode")
out(f"коміт: {commit}, git_dirty: {dirty}")

device = get_device()
model, opt = build_model("frozen", 1, device)
train_loader, val_loader = make_loaders(args.data)  # після build_model, як у train.py
y_val = np.array([y for _, y in val_loader.dataset.items])
ce = torch.nn.CrossEntropyLoss()
ref = json.loads(Path("runs/frozen_seed1/run.json").read_text(encoding="utf-8"))["history"]

out("")
out("| Епоха | Втрата (помилка) | F1 (помилка) | Втрата (правильно) | F1 (правильно) |")
out("|---|---|---|---|---|")
for epoch in range(1, 9):
    model.train()  # навмисна помилка: BN кістяка оновлює статистики
    total = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = ce(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    f1 = f1_score(y_val, predict_logits(model, val_loader, device).argmax(1), average="macro")
    r = ref[epoch - 1]
    out(f"| {epoch} | {total / 1200:.4f} | {f1:.4f} | {r['train_loss']:.4f} | {r['val_macro_f1']:.4f} |")

Path("bn_bug_curve_output.txt").write_text("\n".join(OUT), encoding="utf-8")