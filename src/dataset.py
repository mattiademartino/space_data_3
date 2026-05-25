import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


def _to_float32(arr: np.ndarray) -> np.ndarray:
    data = arr.astype(np.float32)
    if data.max() > 2.0:
        data /= 255.0
    return data


class LunarDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray):
        self.noisy = torch.from_numpy(_to_float32(noisy)).unsqueeze(1)
        self.clean = torch.from_numpy(_to_float32(clean)).unsqueeze(1)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


class LunarTestDataset(Dataset):
    """Noisy-only dataset for blind inference (no clean target available)."""
    def __init__(self, noisy: np.ndarray):
        self.noisy = torch.from_numpy(_to_float32(noisy)).unsqueeze(1)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx]


def load_data(data_cfg: dict, project_root: Path, seed: int = 42):
    noisy = np.load(project_root / data_cfg["noisy_path"])
    clean = np.load(project_root / data_cfg["clean_path"])
    dataset = LunarDataset(noisy, clean)

    val_split = float(data_cfg.get("val_split", 0.05))
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    return train_ds, val_ds


def load_test(data_cfg: dict, project_root: Path) -> LunarTestDataset | None:
    """Load the blind test set if a path is provided."""
    path_key = "test_noisy_path"
    if path_key not in data_cfg:
        return None
    return LunarTestDataset(np.load(project_root / data_cfg[path_key]))


def make_loaders(train_ds, val_ds, batch_size: int, num_workers: int = 4):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
