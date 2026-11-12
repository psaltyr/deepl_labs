import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import f1_score

from checks import run_all_checks
from data import IMG_SIZE, MEAN, STD, make_loaders
from model import (EVAL_MODULES, LR_HEAD, LR_LAYER4, WEIGHT_DECAY, build_model,
                   get_device, set_train_mode)
from split import CLASSES, build_split, check_split, save_split

RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
EPOCHS = 8


def git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
        return commit, dirty
    except Exception:
        return None, None


def device_name(device):
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return device if device == "mps" else (platform.processor() or "cpu")


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    return torch.cat([model(x.to(device)).cpu() for x, _ in loader]).numpy()


def run_one(regime, seed, data_root, device, out_root, epochs):
    out_dir = out_root / f"{regime}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, opt = build_model(regime, seed, device)
    # loader створюється ПІСЛЯ build_model: shuffle бере глобальний генератор, засіяний у кроці 1
    train_loader, val_loader = make_loaders(data_root)
    y_val = np.array([y for _, y in val_loader.dataset.items])
    ce = torch.nn.CrossEntropyLoss()
    n_train = len(train_loader.dataset)

    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        set_train_mode(model, regime)
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        logits = predict_logits(model, val_loader, device)
        f1 = f1_score(y_val, logits.argmax(1), average="macro")
        history.append({"epoch": epoch, "train_loss": total / n_train, "val_macro_f1": float(f1)})
        print(f"{regime} s{seed} epoch {epoch}: loss={total / n_train:.4f} F1={f1:.4f}", flush=True)
    train_time = time.time() - t0

    np.save(out_dir / "val_logits.npy", logits)  # логіти після 8-ї епохи
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "weights.pt")

    commit, dirty = git_info()
    log = {
        "regime": regime, "seed": seed,
        "settings": {
            "epochs": epochs, "steps_per_epoch": len(train_loader),
            "transforms": {"train": f"RandomResizedCrop({IMG_SIZE}, scale=(0.7, 1.0)), RandomHorizontalFlip(), ToTensor(), Normalize",
                           "eval": f"Resize(({IMG_SIZE}, {IMG_SIZE})), ToTensor(), Normalize",
                           "mean": MEAN, "std": STD},
            "model": "resnet18, fc=Linear(512, 6)",
            "weights": None if regime == "scratch" else "IMAGENET1K_V1",
            "eval_modules": EVAL_MODULES[regime],
            "optimizer": {"name": "AdamW", "weight_decay": WEIGHT_DECAY,
                          "lr_head": LR_HEAD, "lr_layer4": LR_LAYER4 if regime == "unfreeze" else None,
                          "betas": [0.9, 0.999], "eps": 1e-8},
            "loss": "CrossEntropyLoss(mean)",
            "train_loader": {"batch_size": 32, "shuffle": True, "drop_last": False, "num_workers": 0},
            "eval_loader": {"batch_size": 64, "shuffle": False, "num_workers": 0},
            "bn": {"momentum": 0.1, "eps": 1e-5},
        },
        "git_commit": commit, "git_dirty": dirty,
        "split_sha256": hashlib.sha256(Path("split.csv").read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "device": device_name(device),
        "history": history, "train_time_s": train_time,
    }
    (out_dir / "run.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def select_and_save(out_root):
    """Вибір лише за validation: режим із найбільшим середнім F1, у ньому найкращий запуск."""
    final = {}
    for regime, seed in RUNS:
        log = json.loads((out_root / f"{regime}_seed{seed}" / "run.json").read_text(encoding="utf-8"))
        final[(regime, seed)] = log["history"][-1]["val_macro_f1"]
    mean = {r: np.mean([f for (rr, _), f in final.items() if rr == r])
            for r in ("scratch", "frozen", "unfreeze")}
    regime = max(mean, key=mean.get)
    seed = max((s for r, s in RUNS if r == regime), key=lambda s: final[(regime, s)])
    state = torch.load(out_root / f"{regime}_seed{seed}" / "weights.pt", map_location="cpu")
    torch.save({"state_dict": state, "regime": regime, "seed": seed,
                "classes": CLASSES, "img_size": IMG_SIZE}, "lab3_model.pt")
    print("середні F1:", {k: round(float(v), 4) for k, v in mean.items()})
    print(f"вибрано: {regime}, seed {seed} -> lab3_model.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="налагоджувальний прогін: 1 епоха, runs_debug/, без вибору моделі")
    args = ap.parse_args()
    device = get_device()
    print("пристрій:", device)

    rows = build_split(Path(args.data))
    save_split(rows)
    if not check_split(rows):
        raise SystemExit("Поділ не збігається з контрольною таблицею")

    run_all_checks(args.data, device)

    out_root = Path("runs_debug" if args.smoke else "runs")
    epochs = 1 if args.smoke else EPOCHS
    for regime, seed in RUNS:
        run_one(regime, seed, args.data, device, out_root, epochs)
    if not args.smoke:
        select_and_save(out_root)