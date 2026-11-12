import argparse
from itertools import islice

import torch

from data import make_loaders, save_batch_figure
from model import (EVAL_MODULES, REGIMES, build_model, count_params,
                   get_device, set_train_mode)
from split import check_split, load_split

MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]


def check_initial_loss(data_root, device):
    print("\n== Початкова втрата (seed 1, перший батч, без кроків оптимізатора) ==")
    ce = torch.nn.CrossEntropyLoss()
    for regime in ("scratch", "frozen"):
        model, _ = build_model(regime, 1, device)
        train_loader, _ = make_loaders(data_root)
        set_train_mode(model, regime)
        with torch.no_grad():
            x, y = next(iter(train_loader))
            loss = ce(model(x.to(device)), y.to(device)).item()
        print(f"{regime:9s} L0 = {loss:.4f}   (ln 6 = {torch.log(torch.tensor(6.0)).item():.4f})")


def check_params():
    print("\n== Параметри ==")
    for regime in REGIMES:
        model, opt = build_model(regime, 1, "cpu")
        total = count_params(model.parameters())
        trainable = count_params(p for p in model.parameters() if p.requires_grad)
        groups = [(count_params(g["params"]), g["lr"]) for g in opt.param_groups]
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        leak = any(id(p) in in_opt for p in model.parameters() if not p.requires_grad)
        print(f"{regime:9s} всього={total} навчається={trainable} групи={groups} "
              f"заморожені в оптимізаторі: {'ТАК (ПОМИЛКА)' if leak else 'ні'}")


def run_steps(regime, mode_fn, data_root, device, n_steps=5):
    model, opt = build_model(regime, 1, device)
    train_loader, _ = make_loaders(data_root)
    snap = lambda: {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    before = snap()
    mode_fn(model)
    ce = torch.nn.CrossEntropyLoss()
    for x, y in islice(train_loader, n_steps):
        opt.zero_grad()
        loss = ce(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
    return model, before, snap()


def module_report(regime, model, before, after):
    param_names = {k for k, _ in model.named_parameters()}
    print(f"{'режим':9s} {'модуль':7s} {'параметри':>10s} {'BN-статистики':>14s} {'очікувано':>10s}")
    all_ok = True
    for m in MODULES:
        keys = [k for k in before if k.split(".")[0] == m]
        pkeys = [k for k in keys if k in param_names]
        bkeys = [k for k in keys if k not in param_names]
        p_changed = any(not torch.equal(before[k], after[k]) for k in pkeys)
        s_changed = (any(not torch.equal(before[k], after[k]) for k in bkeys)
                     if bkeys else None)
        frozen = m in EVAL_MODULES[regime]
        exp_p = not frozen
        exp_s = None if not bkeys else (not frozen)
        ok = (p_changed == exp_p) and (s_changed == exp_s)
        all_ok &= ok
        fmt = lambda v: "немає BN" if v is None else ("так" if v else "ні")
        print(f"{regime:9s} {m:7s} {fmt(p_changed):>10s} {fmt(s_changed):>14s} "
              f"{'так' if ok else 'НІ':>10s}")
    return all_ok


def check_changes(data_root, device):
    print("\n== Що змінюється за 5 кроків (set_train_mode) ==")
    for regime in ("frozen", "unfreeze"):
        model, b, a = run_steps(regime, lambda m, r=regime: set_train_mode(m, r),
                                data_root, device)
        ok = module_report(regime, model, b, a)
        print("збігається з очікуваним:", "ТАК" if ok else "НІ")


def check_bn_experiment(data_root, device):
    print("\n== Дослід з помилкою: frozen + звичайний model.train() ==")
    model, b, a = run_steps("frozen", lambda m: m.train(), data_root, device)
    module_report("frozen", model, b, a)   # тут очікуються розбіжності з очікуваним
    for k in ("layer1.0.bn1.running_mean", "layer1.0.bn1.running_var"):
        print(f"макс. |зміна| {k}: {(a[k] - b[k]).abs().max().item():.6g}")


def run_all_checks(data_root, device):
    print("== Контроль поділу ==")
    print("OK" if check_split(load_split()) else "ПОМИЛКА")
    train_loader, _ = make_loaders(data_root)
    torch.manual_seed(1)
    save_batch_figure(train_loader)
    check_initial_loss(data_root, device)
    check_params()
    check_changes(data_root, device)
    check_bn_experiment(data_root, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    device = get_device()
    print("пристрій:", device)
    run_all_checks(args.data, device)