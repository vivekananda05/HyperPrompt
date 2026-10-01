# dataset.py

import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from hsi_preprocessing import preprocess_hsi_dataset


# ---------------------------------------------------
# Reproducibility
# ---------------------------------------------------
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------
# Augmentations
# ---------------------------------------------------
def flip_augmentation(x):
    if torch.rand(1) > 0.5:
        x = torch.flip(x, dims=[2])
    if torch.rand(1) > 0.5:
        x = torch.flip(x, dims=[3])
    return x


def radiation_augmentation(x, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = torch.empty((x.shape[0], 1, 1, 1),
                        device=x.device).uniform_(*alpha_range)
    noise = torch.randn_like(x)
    return alpha * x + beta * noise


def mixture_augmentation(x, dataset, label, beta=1/25):
    same_class_indices = dataset.class_indices[int(label)]
    if len(same_class_indices) < 2:
        return x

    idx = random.choice(same_class_indices)
    x2 = torch.tensor(dataset.patches[idx],
                      dtype=torch.float32,
                      device=x.device).unsqueeze(0)

    alpha1, alpha2 = torch.rand(2, device=x.device)
    noise = torch.randn_like(x)

    return (alpha1 * x + alpha2 * x2) / (alpha1 + alpha2 + 1e-6) + beta * noise





# ---------------------------------------------------
# Dataset
# ---------------------------------------------------
class HSIPatchDataset(Dataset):

    def __init__(self,
                 hsi_path,
                 gt_path,
                 patch_size=13,
                 stride=1,
                 flip_aug=False,
                 radiation_aug=False,
                 mixture_aug=False,
                 ignored_labels=None,
                 re_ratio=1,
                 split_ratio=1.0,
                 split='train',
                 seed=42):

        set_random_seed(seed)

        self.patches, self.labels, self.locations = preprocess_hsi_dataset(
            hsi_path,
            gt_path,
            patch_size=patch_size,
            stride=stride,
            ignored_labels=ignored_labels
        )

        self.patches = np.transpose(self.patches, (0, 3, 1, 2))

        # -------------------------------------------------
        # Train/Test Split
        # -------------------------------------------------
        num_samples = len(self.labels)
        indices = np.arange(num_samples)
        np.random.RandomState(seed).shuffle(indices)

        split_idx = int(num_samples * split_ratio)

        if split == 'train':
            split_indices = indices[:split_idx]
        elif split == 'test':
            split_indices = indices[split_idx:]
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.patches = self.patches[split_indices]
        self.labels = self.labels[split_indices]
        self.locations = self.locations[split_indices]

        self.flip_aug = flip_aug
        self.radiation_aug = radiation_aug
        self.mixture_aug = mixture_aug

        self.re_ratio = max(1, int(re_ratio)) if split == 'train' else 1
        self.num_samples = len(self.labels)

        # Build class index map
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[int(label)].append(idx)

    def __len__(self):
        return self.num_samples * self.re_ratio

    def __getitem__(self, idx):

        real_idx = idx % self.num_samples
        patch = torch.tensor(self.patches[real_idx],
                             dtype=torch.float32).unsqueeze(0)

        label = self.labels[real_idx]

        if self.flip_aug:
            patch = flip_augmentation(patch)

        if self.radiation_aug and random.random() < 0.5:
            patch = radiation_augmentation(patch)

        if self.mixture_aug and random.random() < 0.5:
            patch = mixture_augmentation(patch, self, label)

        patch = patch.squeeze(0)

        return {
            "image": patch,
            "label": torch.tensor(label, dtype=torch.long),
            "location": torch.tensor(self.locations[real_idx],
                                     dtype=torch.long)
        }


# ---------------------------------------------------
# CUDA Prefetcher
# ---------------------------------------------------
class CUDAPrefetcher:

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device=device)
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            for k in self.next_batch:
                self.next_batch[k] = self.next_batch[k].to(
                    self.device, non_blocking=True
                )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            return None
        self.preload()
        return batch


# ---------------------------------------------------
# Dataloader Builder
# ---------------------------------------------------
def build_dataloader(hsi_path,
                     gt_path,
                     batch_size,
                     patch_size=13,
                     stride=1,
                     flip_aug=False,
                     radiation_aug=False,
                     mixture_aug=False,
                     ignored_labels=None,
                     re_ratio=1,
                     split_ratio=1.0,
                     split='train',
                     shuffle=True,
                     num_workers=0,
                     seed=42):

    dataset = HSIPatchDataset(
        hsi_path,
        gt_path,
        patch_size,
        stride,
        flip_aug,
        radiation_aug,
        mixture_aug,
        ignored_labels,
        re_ratio,
        split_ratio,
        split,
        seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader
