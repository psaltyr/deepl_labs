from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions, split_summary

FN_COST = 10
FP_COST = 1


def confusion(y: np.ndarray, s: np.ndarray, t: float) -> tuple[int, int, int, int]:
    pred = s >= t
    pos = y == 1
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    tn = int(np.sum(~pred & ~pos))
    return tp, fp, fn, tn


def cost(fn: int, fp: int) -> int:
    return FN_COST * fn + FP_COST * fp


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def fast_candidates(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]

    p = int(np.sum(y == 1))
    n = len(y) - p
    th = np.r_[s_sorted[last], np.inf]
    tp = np.r_[tp_cum[last], 0]
    fp = np.r_[fp_cum[last], 0]
    fn = p - tp
    tn = n - fp
    table = pd.DataFrame(
        {"threshold": th, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": FN_COST * fn + FP_COST * fp}
    )
    return table.sort_values("threshold").reset_index(drop=True)


def best_threshold(table: pd.DataFrame) -> float:
    return float(table.loc[table["C"] == table["C"].min(), "threshold"].max())


def slow_candidates(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    rows = []
    for t in list(np.unique(s)) + [np.inf]:
        tp, fp, fn, tn = confusion(y, s, float(t))
        rows.append((float(t), tp, fp, fn, tn, cost(fn, fp)))
    table = pd.DataFrame(rows, columns=["threshold", "TP", "FP", "FN", "TN", "C"])
    return table.sort_values("threshold").reset_index(drop=True)


def slow_best(table: pd.DataFrame) -> float:
    min_c = min(table["C"])
    return float(max(t for t, c in zip(table["threshold"], table["C"]) if c == min_c))


CHECKS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0], 0.4),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0], float("inf")),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1] + [1] + [0] * 10 + [0], 0.8),
}


def run_checks(out: Path) -> None:
    for name, (s, y, expected_t) in CHECKS.items():
        s = np.array(s, dtype=np.float64)
        y = np.array(y, dtype=np.int64)
        slow = slow_candidates(y, s)
        fast = fast_candidates(y, s)
        same = slow.shape == fast.shape and bool((slow.to_numpy() == fast.to_numpy()).all())
        t_slow, t_fast = slow_best(slow), best_threshold(fast)

        print(f"\nНабір {name}")
        print(slow.to_string(index=False))
        print(f"  таблиці збігаються: {same}")
        print(f"  t* повільний = {t_slow}, швидкий = {t_fast}, очікуваний = {expected_t}")
        slow.to_csv(out / f"check_{name}.csv", index=False)

        if not (same and t_slow == t_fast == expected_t):
            raise SystemExit(f"Перевірка {name} НЕ пройдена. До validation/test не переходимо.")
    print("\nВсі три перевірки пройдено.\n")

def save_threshold(out: Path, t: float) -> None:
    value = t if np.isfinite(t) else "inf"  
    (out / "threshold.json").write_text(
        json.dumps({"t_star": value}, indent=2) + "\n", encoding="utf-8"
    )


def load_threshold(out: Path) -> float:
    return float(json.loads((out / "threshold.json").read_text(encoding="utf-8"))["t_star"])


def analyze_validation(out: Path) -> float:
    data = load_predictions(out)
    y = data["validation_y"]
    s = data["validation_scores"].astype(np.float64)  

    table = fast_candidates(y, s)
    table.to_csv(out / "validation_candidates.csv", index=False)
    t_star = best_threshold(table)

    rows = []
    for name, t in [("Стандартний поріг", 0.5), ("Мінімум вартості", t_star), ("Усі негативні", np.inf)]:
        tp, fp, fn, tn = confusion(y, s, t)
        rows.append(
            {
                "rule": name,
                "t": t,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "C": cost(fn, fp),
                "precision": safe_div(tp, tp + fp),
                "recall": safe_div(tp, tp + fn),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "validation_summary.csv", index=False)
    print("VALIDATION")
    print(summary.to_string(index=False))

    auc = roc_auc_score(y, s)
    print(f"ROC-AUC (validation): {auc:.6f}")
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR, validation")
    plt.savefig(out / "pr_validation.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")
    plt.title(f"ROC, validation, AUC={auc:.4f}")
    plt.savefig(out / "roc_validation.png", dpi=150)
    plt.close()

    save_threshold(out, t_star)
    print(f"Зафіксовано t* = {t_star!r}\n")
    return t_star


def analyze_test(out: Path) -> None:
    t = load_threshold(out)  
    data = load_predictions(out)
    y = data["test_y"]
    s = data["test_scores"].astype(np.float64)
    amount = data["test_amount"]
    row_ids = data["test_row_ids"]

    tp, fp, fn, tn = confusion(y, s, t)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    result = pd.DataFrame(
        [{"t": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp),
          "precision": precision, "recall": recall}]
    )
    result.to_csv(out / "test_summary.csv", index=False)
    print("TEST")
    print(result.to_string(index=False))
    if np.isnan(precision):
        print("precision не визначена: TP + FP = 0 (жодного позитивного прогнозу).")
    if np.isnan(recall):
        print("recall не визначений: TP + FN = 0 (у test немає шахрайських транзакцій).")

    pred = s >= t
    order = np.argsort(row_ids)  
    rows = []
    for kind, mask in [("FP", pred & (y == 0)), ("FN", ~pred & (y == 1))]:
        for i in [i for i in order if mask[i]][:5]:
            rows.append(
                {
                    "csv_row": int(row_ids[i]),
                    "type": kind,
                    "label": int(y[i]),
                    "score": float(s[i]),
                    "amount": float(amount[i]),
                    "abs_s_minus_t": abs(float(s[i]) - t),
                }
            )
    errors = pd.DataFrame(rows)
    errors.to_csv(out / "test_errors.csv", index=False)
    print("\nПерші помилки за номером рядка CSV:")
    print(errors.to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved", type=Path, required=True, help="Тека зі збереженими прогнозами (results)")
    args = parser.parse_args()
    out: Path = args.saved
    if not (out / "predictions.npz").exists():
        parser.error(f"У теці {out} немає predictions.npz. Спочатку запустіть навчання.")

    print(split_summary(load_predictions(out)).to_string(index=False))
    run_checks(out)
    analyze_validation(out)
    analyze_test(out)


if __name__ == "__main__":
    main()