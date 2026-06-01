"""Training and evaluation loop for the Conditional DDPM denoiser."""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nvidia_loss import MixLoss   # noqa: E402


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def build_criterion(loss_name: str) -> nn.Module:
    """
    l1  : plain L1 — fast, good for high-noise regimes.
    mse : MSE — tends to over-smooth (kept for ablation).
    mix : MS-SSIM + L1 (Zhao et al. 2017, α=0.84) — best perceptual quality.
    """
    loss_name = loss_name.lower()
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mix":
        return MixLoss(alpha=0.84, data_range=1.0)
    raise ValueError(f"Unknown diffusion loss: {loss_name!r}")


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate(diffusion_model, val_loader, device, n_ddim_steps: int = 20):
    """
    Run DDIM inference on the full validation set and return MSE and PSNR.
    Uses fewer DDIM steps than final inference for speed.
    """
    diffusion_model.model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for noisy_b, clean_b in val_loader:
            pred = diffusion_model.ddim_sample(noisy_b.to(device), n_steps=n_ddim_steps).cpu()
            all_preds.append(pred)
            all_targets.append(clean_b)

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mse  = torch.mean((preds - targets) ** 2).item()
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    return mse, psnr


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def fit_diffusion(
    diffusion_model,
    train_loader,
    val_loader,
    cfg:        dict,
    device:     torch.device,
    output_dir: Path,
    run_name:   str,
) -> dict:
    """
    Train the conditional DDPM.

    At each training step:
      - sample a random timestep t ~ Uniform[0, T) for each image in the batch
      - corrupt x0 → x_t via the forward process
      - ask the model to predict x0 from (x_t, y, t)
      - back-propagate the chosen pixel-space loss on x0_pred vs x0

    Validation is run every `val_every` epochs using DDIM with a small number
    of steps (faster than full inference) to track PSNR.
    """
    epochs     = int(cfg.get("epochs", 100))
    lr         = float(cfg.get("lr", 5e-4))
    wd         = float(cfg.get("weight_decay", 1e-4))
    loss_name  = cfg.get("loss", "l1").lower()
    val_steps  = int(cfg.get("val_ddim_steps", 20))
    val_every  = int(cfg.get("val_every", 5))

    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = build_criterion(loss_name)
    optimizer = optim.AdamW(diffusion_model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses: list[float]             = []
    val_records:  list[tuple[int, float]] = []   # (epoch, psnr)
    best_psnr    = -float("inf")
    best_val_mse = float("inf")

    for epoch in range(1, epochs + 1):
        diffusion_model.model.train()
        total, count = 0.0, 0

        for noisy_b, clean_b in train_loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            loss = diffusion_model.training_step(clean_b, noisy_b, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * noisy_b.size(0)
            count += noisy_b.size(0)

        train_loss = total / count
        train_losses.append(train_loss)
        scheduler.step()

        # Validate periodically (DDIM is costlier than a single forward pass)
        do_val = (epoch % val_every == 0) or (epoch == 1) or (epoch == epochs)
        if do_val:
            val_mse, val_psnr = validate(diffusion_model, val_loader, device, val_steps)
            val_records.append((epoch, val_psnr))

            if val_psnr > best_psnr:
                best_psnr    = val_psnr
                best_val_mse = val_mse
                torch.save(diffusion_model.state_dict(), output_dir / "best_model.pt")

            print(f"  [{run_name}] {epoch:3d}/{epochs}  "
                  f"train {train_loss:.5f}  val_psnr {val_psnr:.2f} dB")
        else:
            if epoch % 10 == 0:
                print(f"  [{run_name}] {epoch:3d}/{epochs}  train {train_loss:.5f}")

    torch.save(diffusion_model.state_dict(), output_dir / "final_model.pt")
    _save_plots(train_losses, val_records, epochs, run_name, output_dir)

    return {
        "train_losses":  train_losses,
        "val_psnrs":     val_records,
        "best_val_psnr": best_psnr,
        "best_val_mse":  best_val_mse,
    }


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def _save_plots(train_losses, val_records, epochs, name, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, epochs + 1), train_losses, color="steelblue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title(f"Training Loss — {name}")
    ax1.grid(True, alpha=0.3)

    if val_records:
        ep, psnrs = zip(*val_records)
        ax2.plot(ep, psnrs, marker="o", color="tomato")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("PSNR (dB)")
        ax2.set_title(f"Validation PSNR (DDIM) — {name}")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=150)
    plt.close()
