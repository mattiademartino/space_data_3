from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import nvidia_loss 


def build_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    name = cfg.get("optimizer", "adam").lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 1e-4))
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                         momentum=float(cfg.get("momentum", 0.9)))
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(optimizer: optim.Optimizer, cfg: dict, epochs: int):
    name = cfg.get("scheduler", "cosine").lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", max(1, epochs // 3))),
            gamma=float(cfg.get("gamma", 0.1)),
        )
    elif name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name!r}")


def build_criterion(cfg: dict) -> nn.Module:
    loss = cfg.get("loss", "mse").lower()
    if loss == "mse":
        return nn.MSELoss()
    elif loss == "l1":
        return nn.L1Loss()
    elif loss == "combined":
        _mse, _l1 = nn.MSELoss(), nn.L1Loss()
    elif loss == "nvidia_loss":
        return nvidia_loss.MixLoss()

        class _Combined(nn.Module):
            def forward(self, pred, target):
                return 0.5 * _mse(pred, target) + 0.5 * _l1(pred, target)

        return _Combined()
    raise ValueError(f"Unknown loss: {loss!r}")


def _run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total, count = 0.0, 0

    if is_train:
        for noisy_b, clean_b in loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            pred = model(noisy_b)
            loss = criterion(pred, clean_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * noisy_b.size(0)
            count += noisy_b.size(0)
    else:
        with torch.no_grad():
            for noisy_b, clean_b in loader:
                noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
                loss = criterion(model(noisy_b), clean_b)
                total += loss.item() * noisy_b.size(0)
                count += noisy_b.size(0)

    return total / count


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    train_cfg: dict,
    device: torch.device,
    output_dir: Path,
    run_name: str,
) -> dict:
    epochs = int(train_cfg.get("epochs", 60))
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, epochs)
    criterion = build_criterion(train_cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, criterion, device, optimizer)
        val = _run_epoch(model, val_loader, criterion, device)
        train_losses.append(tr)
        val_losses.append(val)

        if scheduler is not None:
            scheduler.step()

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{run_name}] {epoch:3d}/{epochs}  "
                  f"train {tr:.6f}  val {val:.6f}")

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    _save_loss_plot(train_losses, val_losses, epochs, run_name, output_dir)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
    }


def _save_loss_plot(train_losses, val_losses, epochs, name, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(1, epochs + 1)
    ax.plot(x, train_losses, label="Train")
    ax.plot(x, val_losses, label="Validation", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves — {name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close()
