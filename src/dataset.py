import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class LunarDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray):
        self.noisy = torch.tensor(noisy / 255.0, dtype=torch.float32).unsqueeze(1)
        self.clean = torch.tensor(clean / 255.0, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


def load_data(data_cfg: dict, project_root: Path, seed: int = 42):
    noisy = np.load(project_root / data_cfg["noisy_path"])
    clean = np.load(project_root / data_cfg["clean_path"])
    dataset = LunarDataset(noisy, clean)

    val_size = int(data_cfg["val_split"] * len(dataset))
    train_size = len(dataset) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
    return train_ds, val_ds


def make_loaders(train_ds, val_ds, batch_size: int, num_workers: int = 2):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
