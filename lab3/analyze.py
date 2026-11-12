import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from data import IMG_SIZE, SplitDataset, eval_transform
from split import CLASSES, load_split
from train import RUNS

sys.stdout.reconfigure(encoding="utf-8")
OUT = []


def out(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    OUT.append(s)


def load_logs(runs_dir):
    return {(r, s): json.loads((runs_dir / f"{r}_seed{s}" / "run.json").read_text(encoding="utf-8"))
            for r, s in RUNS}


def check_hash(logs):
    current = hashlib.sha256(Path("split.csv").read_bytes()).hexdigest()
    bad = [k for k, l in logs.items() if l["split_sha256"] != current]
    if bad:
        raise SystemExit(f"Хеш split.csv у журналах {bad} не збігається з поточним файлом")
    out("Хеш split.csv однаковий у всіх 7 журналах і збігається з поточним:", current)
    commits = sorted({(l["git_commit"], l["git_dirty"]) for l in logs.values()}, key=str)
    out("Коміти / git_dirty у журналах:", commits)


def check_val_logits(logs, runs_dir):
    y_val = np.array([CLASSES.index(l) for _, l, p in load_split() if p == "validation"])
    for (r, s), log in logs.items():
        logits = np.load(runs_dir / f"{r}_seed{s}" / "val_logits.npy")
        f1 = f1_score(y_val, logits.argmax(1), average="macro")
        if abs(f1 - log["history"][-1]["val_macro_f1"]) > 1e-6:
            raise SystemExit(f"val_logits.npy не відповідає журналу для {r} seed {s}")
    out("val_logits.npy відповідає F1 у журналах для всіх 7 запусків")


def runs_table(logs):
    out("\n| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с |")
    out("|---|---|---|---|---|---|")
    for i, (r, s) in enumerate(RUNS, 1):
        l = logs[(r, s)]
        h = l["history"][-1]
        out(f"| {i} | {r} | {s} | {h['train_loss']:.4f} | {h['val_macro_f1']:.4f} | {l['train_time_s']:.0f} |")


def curves_figure(logs):
    colors = {"scratch": "tab:red", "frozen": "tab:blue", "unfreeze": "tab:green"}
    styles = {1: "-", 2: "--", 3: ":"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for (r, s), l in logs.items():
        ep = [h["epoch"] for h in l["history"]]
        kw = dict(color=colors[r], linestyle=styles[s], marker="o", markersize=3, label=f"{r} s{s}")
        ax[0].plot(ep, [h["train_loss"] for h in l["history"]], **kw)
        ax[1].plot(ep, [h["val_macro_f1"] for h in l["history"]], **kw)
    ax[0].set_title("Втрата на train")
    ax[1].set_title("macro-F1 на validation")
    for a in ax:
        a.set_xlabel("Епоха")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Втрата")
    ax[1].set_ylabel("macro-F1")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("curves.png", dpi=150)
    plt.close(fig)


def effects(logs):
    final = {k: l["history"][-1]["val_macro_f1"] for k, l in logs.items()}
    vals = {r: [final[(rr, s)] for rr, s in RUNS if rr == r] for r in ("scratch", "frozen", "unfreeze")}
    mean = {r: float(np.mean(v)) for r, v in vals.items()}
    rng = {r: max(v) - min(v) for r, v in vals.items() if r != "scratch"}
    R = max(rng["frozen"], rng["unfreeze"])
    d_pre = mean["frozen"] - vals["scratch"][0]
    d_unf = mean["unfreeze"] - mean["frozen"]
    out("\n| Величина | Значення |")
    out("|---|---|")
    out(f"| F_scratch,1 | {vals['scratch'][0]:.4f} |")
    out(f"| F̄_frozen | {mean['frozen']:.4f} |")
    out(f"| F̄_unfreeze | {mean['unfreeze']:.4f} |")
    out(f"| R_frozen | {rng['frozen']:.4f} |")
    out(f"| R_unfreeze | {rng['unfreeze']:.4f} |")
    out(f"| R = max(R_frozen, R_unfreeze) | {R:.4f} |")
    out(f"| Δ_pre | {d_pre:+.4f} |")
    out(f"| Δ_unf | {d_unf:+.4f} |")
    for name, d in (("Δ_pre", d_pre), ("Δ_unf", d_unf)):
        if abs(d) <= R:
            v = "|Δ| ≤ R: різниця не відрізняється від розкиду, результату немає"
        else:
            v = "|Δ| > R: ефект вартий перевірки, але не доведений"
        out(f"{name}: |Δ|/R = {abs(d) / R:.2f} -> {v}")
    return final, mean, R, d_unf


def check_selection(final, mean, R, d_unf):
    regime = max(mean, key=mean.get)
    seed = max((s for r, s in RUNS if r == regime), key=lambda s: final[(regime, s)])
    ckpt = torch.load("lab3_model.pt", map_location="cpu")
    if ckpt["regime"] != regime or ckpt["seed"] != seed:
        raise SystemExit(f"lab3_model.pt ({ckpt['regime']}, seed {ckpt['seed']}) не відповідає "
                         f"правилу вибору ({regime}, seed {seed})")
    out(f"\nВибір за validation: {regime}, seed {seed}; lab3_model.pt відповідає правилу")
    if abs(d_unf) <= R:
        out("Увага: |Δ_unf| ≤ R, тож вибір між frozen і unfreeze визначив шум")
    return ckpt, final[(regime, seed)]


def test_evaluation(ckpt, val_f1, data_root):
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()  # CPU

    ds = SplitDataset(data_root, "test", eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    with torch.no_grad():
        logits = torch.cat([model(x) for x, _ in loader]).numpy()
    y = np.array([lab for _, lab in ds.items])
    pred = logits.argmax(1)
    n = len(y)
    f1 = f1_score(y, pred, average="macro")
    acc = float((pred == y).mean())
    se = float(np.sqrt(acc * (1 - acc) / n))
    out(f"\nTest: n = {n}, macro-F1 = {f1:.4f}, accuracy = {acc:.4f}")
    out(f"macro-F1 на validation вибраного запуску = {val_f1:.4f}; різниця test - validation = {f1 - val_f1:+.4f}")
    out(f"Стандартна похибка частки правильних sqrt(p(1-p)/n) = {se:.4f}")

    cm = confusion_matrix(y, pred, labels=list(range(6)))
    out("\nМатриця плутанини (рядки - справжній клас, стовпці - передбачений):\n")
    out("| справжній \\ передбачений | " + " | ".join(CLASSES) + " |")
    out("|---|" + "---|" * 6)
    for i, c in enumerate(CLASSES):
        out(f"| {c} | " + " | ".join(str(v) for v in cm[i]) + " |")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels(CLASSES)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Передбачений клас")
    ax.set_ylabel("Справжній клас")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    plt.close(fig)

    # п'ять характерних помилок
    pairs = [(t, p, int(cm[t, p])) for t in range(6) for p in range(6) if t != p and cm[t, p] > 0]
    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    names = [Path(path).name for path, _ in ds.items]
    cands = {(t, p): sorted(names[i] for i in range(n) if y[i] == t and pred[i] == p)
             for t, p, _ in pairs}
    chosen, rnd = [], 0
    while len(chosen) < 5 and rnd < 5:
        for t, p, c in pairs:
            if len(chosen) == 5:
                break
            if len(cands[(t, p)]) > rnd:
                chosen.append((cands[(t, p)][rnd], t, p, c))
        rnd += 1

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    idx_by_name = {nm: i for i, nm in enumerate(names)}
    out("\n| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | Кількість таких помилок у test |")
    out("|---|---|---|---|---|")
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, (name, t, p, c) in zip(axes, chosen):
        i = idx_by_name[name]
        out(f"| {name} | {CLASSES[t]} | {CLASSES[p]} | {probs[i, p]:.3f} | {c} |")
        ax.imshow(Image.open(ds.items[i][0]).convert("RGB"))
        ax.set_title(f"{name}\n{CLASSES[t]} → {CLASSES[p]}\np = {probs[i, p]:.2f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("errors.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="шлях до набору")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    runs_dir = Path(args.runs)

    logs = load_logs(runs_dir)
    check_hash(logs)
    check_val_logits(logs, runs_dir)
    runs_table(logs)
    curves_figure(logs)
    final, mean, R, d_unf = effects(logs)
    ckpt, val_f1 = check_selection(final, mean, R, d_unf)
    test_evaluation(ckpt, val_f1, args.data)
    Path("analysis_output.txt").write_text("\n".join(OUT), encoding="utf-8")