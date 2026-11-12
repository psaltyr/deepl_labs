import torch
import numpy as np
import torchvision
from torchvision.models import ResNet18_Weights

REGIMES = ("scratch", "frozen", "unfreeze")

# що навчається (імена модулів верхнього рівня)
TRAINABLE = {
    "scratch": None,                       # усе
    "frozen": ["fc"],
    "unfreeze": ["layer4", "fc"],
}
# що в eval() під час навчання
EVAL_MODULES = {
    "scratch": [],
    "frozen": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "unfreeze": ["conv1", "bn1", "layer1", "layer2", "layer3"],
}
LR_HEAD = 3e-3
LR_LAYER4 = 3e-4
WEIGHT_DECAY = 1e-4


def build_model(regime, seed, device):
    """Кроки 1-5 із завдання. Повертає (model, optimizer)."""
    assert regime in REGIMES
    # 1. зерно
    torch.manual_seed(seed)
    np.random.seed(seed)
    # 2. ваги
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    # 3. нова голова
    model.fc = torch.nn.Linear(512, 6)
    # 4. requires_grad
    trainable = TRAINABLE[regime]
    for name, p in model.named_parameters():
        p.requires_grad = True if trainable is None else name.split(".")[0] in trainable
    # 5. пристрій і оптимізатор
    model.to(device)
    optimizer = torch.optim.AdamW(param_groups(model, regime), weight_decay=WEIGHT_DECAY)
    return model, optimizer


def param_groups(model, regime):
    if regime == "scratch":
        return [{"params": list(model.parameters()), "lr": LR_HEAD}]
    if regime == "frozen":
        return [{"params": list(model.fc.parameters()), "lr": LR_HEAD}]
    return [
        {"params": list(model.fc.parameters()), "lr": LR_HEAD},
        {"params": list(model.layer4.parameters()), "lr": LR_LAYER4},
    ]


def set_train_mode(model, regime):
    """Викликати на початку кожної епохи."""
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


def count_params(params):
    return sum(p.numel() for p in params)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":
    for regime in REGIMES:
        model, opt = build_model(regime, 1, "cpu")
        total = count_params(model.parameters())
        trainable = count_params(p for p in model.parameters() if p.requires_grad)
        groups = [(count_params(g["params"]), g["lr"]) for g in opt.param_groups]
        print(regime, "всього:", total, "навчається:", trainable, "групи:", groups)