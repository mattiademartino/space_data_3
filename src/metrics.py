import math
import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse_val = torch.mean((pred - target) ** 2).item()
    if mse_val == 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse_val)


def compute_metrics(model: torch.nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for noisy_b, clean_b in loader:
            pred = model(noisy_b.to(device)).cpu()
            all_preds.append(pred)
            all_targets.append(clean_b)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return {"mse": mse(preds, targets), "psnr": psnr(preds, targets)}
