import argparse
import csv
from pathlib import Path

import numpy as np

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# контрольна таблиця із завдання: (кількість, сума числових імен)
EXPECTED = {
    "buildings": {"train": (200, 1996761), "validation": (218, 4837323), "test": (219, 4837559)},
    "forest":    {"train": (200, 2032625), "validation": (237, 5276962), "test": (237, 5291326)},
    "glacier":   {"train": (200, 2183923), "validation": (276, 6135652), "test": (277, 6157250)},
    "mountain":  {"train": (200, 1950345), "validation": (262, 5789535), "test": (263, 5849765)},
    "sea":       {"train": (200, 1987613), "validation": (255, 5655068), "test": (255, 5668011)},
    "street":    {"train": (200, 1841817), "validation": (250, 5541845), "test": (251, 5547533)},
}


def sorted_files(folder: Path):
    # sorted() за рядками, як у ImageFolder
    return sorted(p.name for p in folder.glob("*.jpg"))


def build_split(data_root: Path):
    train_dir = data_root / "seg_train" / "seg_train"
    test_dir = data_root / "seg_test" / "seg_test"
    rows = []

    rng = np.random.default_rng(0)  # один на всі класи
    for cls in CLASSES:
        files = sorted_files(train_dir / cls)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        for i in idx[:200]:
            rows.append((files[i], cls, "train"))

    rng = np.random.default_rng(0)  # новий, теж один на всі класи
    for cls in CLASSES:
        files = sorted_files(test_dir / cls)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        for i in idx[:cut]:
            rows.append((files[i], cls, "validation"))
        for i in idx[cut:]:
            rows.append((files[i], cls, "test"))
    return rows


def save_split(rows, path="split.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "part"])
        w.writerows(rows)


def load_split(path="split.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["file"], r["label"], r["part"]) for r in csv.DictReader(f)]


def check_split(rows):
    ok = True
    print(f"{'клас':10s} {'part':11s} {'отримано':>20s} {'очікувано':>20s}  збіг")
    for cls in CLASSES:
        for part in ("train", "validation", "test"):
            names = [r[0] for r in rows if r[1] == cls and r[2] == part]
            got = (len(names), sum(int(Path(n).stem) for n in names))
            exp = EXPECTED[cls][part]
            match = got == exp
            ok &= match
            print(f"{cls:10s} {part:11s} {got[0]:>5d} / {got[1]:>12d} "
                  f"{exp[0]:>5d} / {exp[1]:>12d}  {'так' if match else 'НІ'}")
    # жоден файл не входить у дві частини
    names = [r[0] for r in rows]
    disjoint = len(names) == len(set(names))
    print("файли не перетинаються між частинами:", "так" if disjoint else "НІ")
    return ok and disjoint


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="шлях до розпакованого набору")
    args = ap.parse_args()
    rows = build_split(Path(args.data))
    save_split(rows)
    print("всі 18 клітинок і перетин:", "OK" if check_split(rows) else "ПОМИЛКА")