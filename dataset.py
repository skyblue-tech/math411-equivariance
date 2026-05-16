import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


class MaskTransform:
    """Resize a trimap mask and binarize: foreground (label 1) -> 1, else -> 0."""

    def __init__(self, size):
        self.size = size

    def __call__(self, mask):
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        arr = np.array(mask, dtype=np.int64)
        binary = (arr == 1).astype(np.float32)
        return torch.from_numpy(binary).unsqueeze(0)   # 1×H×W


class PetSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="trainval", size=128, download=True):
        img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self._mask_tf = MaskTransform(size)
        self._base = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            transform=img_tf,
            download=download,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        img, mask_pil = self._base[idx]
        return img, self._mask_tf(mask_pil)


def get_loaders(root="data", size=128, batch_size=16, val_fraction=0.2, seed=42):
    full_train = PetSegDataset(root, split="trainval", size=size)
    test_set = PetSegDataset(root, split="test", size=size)

    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    # num_workers=0 avoids Windows multiprocessing spawn pickling issues
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader
