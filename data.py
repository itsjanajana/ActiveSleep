import os
from typing import Callable, Optional, List, Dict

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F


class MultiDomainBreastUSDataset(Dataset):
    """
    Generic multi-domain breast ultrasound dataset.

    Expects a CSV with columns: image_path, label, domain, split
    where:
        - image_path: path relative to root_dir
        - label: string label (e.g., 'benign', 'malignant')
        - domain: string identifier of dataset / scanner (e.g., 'BUSI', 'BUS-BRA')
        - split: 'train', 'val', 'test' (or other names, but consistent)
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        # Label / domain vocabularies
        self.label_names: List[str] = sorted(self.df["label"].unique().tolist())
        self.label2idx: Dict[str, int] = {name: i for i, name in enumerate(self.label_names)}
        self.idx2label: Dict[int, str] = {i: name for name, i in self.label2idx.items()}

        self.domain_names: List[str] = sorted(self.df["domain"].unique().tolist())
        self.domain2idx: Dict[str, int] = {name: i for i, name in enumerate(self.domain_names)}
        self.idx2domain: Dict[int, str] = {i: name for name, i in self.domain2idx.items()}

        self.transform = transform
        self._to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def get_split_indices(self, split_name: str) -> torch.Tensor:
        """Return indices for a given split name."""
        mask = self.df["split"] == split_name
        indices = self.df.index[mask].to_numpy()
        return torch.tensor(indices, dtype=torch.long)

    def get_raw_image(self, idx: int) -> Image.Image:
        row = self.df.iloc[int(idx)]
        img_path = os.path.join(self.root_dir, row["image_path"])
        img = Image.open(img_path).convert("L")  # breast US is typically grayscale
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        img = self.get_raw_image(idx)
        domain_name = row["domain"]
        domain_idx = self.domain2idx[domain_name]

        if self.transform is not None:
            img_tensor = self.transform(img, domain_idx)
        else:
            # Basic fallback: tensor + channel repeat
            img_tensor = self._to_tensor(img)
            img_tensor = img_tensor.repeat(3, 1, 1)

        label_name = row["label"]
        label_idx = self.label2idx[label_name]

        sample = {
            "image": img_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "domain": torch.tensor(domain_idx, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }
        return sample


class DomainHarmonizer:
    """
    Simple domain harmonization via per-domain first/second moment matching
    to a reference domain (e.g., BUSI). Interface is compatible with replacing
    this with a trained diffusion model later.

    Usage:
        harmonizer = DomainHarmonizer(num_domains, idx2domain, img_size=256)
        harmonizer.fit(dataset, split_name='train')
        dataset.transform = harmonizer  # __call__(image, domain_idx) -> tensor
    """

    def __init__(
        self,
        num_domains: int,
        idx2domain: Dict[int, str],
        img_size: int = 256,
        reference_domain_idx: int = 0,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
    ) -> None:
        self.num_domains = num_domains
        self.idx2domain = idx2domain
        self.img_size = img_size
        self.reference_domain_idx = reference_domain_idx

        self._to_tensor = transforms.ToTensor()
        self.normalize_mean = torch.tensor(normalize_mean).view(3, 1, 1)
        self.normalize_std = torch.tensor(normalize_std).view(3, 1, 1)

        self.domain_means = torch.zeros(num_domains, dtype=torch.float32)
        self.domain_stds = torch.ones(num_domains, dtype=torch.float32)
        self._fitted = False

    def fit(self, dataset: MultiDomainBreastUSDataset, split_name: Optional[str] = None) -> None:
        """
        Estimate per-domain mean and std of raw intensities over the given split.
        If split_name is None, use all samples.
        """
        if split_name is None:
            indices = torch.arange(len(dataset), dtype=torch.long)
        else:
            indices = dataset.get_split_indices(split_name)

        sum_intensity = torch.zeros(self.num_domains, dtype=torch.float64)
        sum_sq_intensity = torch.zeros(self.num_domains, dtype=torch.float64)
        count_intensity = torch.zeros(self.num_domains, dtype=torch.float64)

        for idx in indices.tolist():
            row = dataset.df.iloc[int(idx)]
            domain_name = row["domain"]
            domain_idx = dataset.domain2idx[domain_name]

            img = dataset.get_raw_image(idx)
            x = self._to_tensor(img).view(-1).double()

            sum_intensity[domain_idx] += x.sum()
            sum_sq_intensity[domain_idx] += (x * x).sum()
            count_intensity[domain_idx] += x.numel()

        for d in range(self.num_domains):
            if count_intensity[d] > 0:
                mean = sum_intensity[d] / count_intensity[d]
                var = sum_sq_intensity[d] / count_intensity[d] - mean * mean
                std = torch.sqrt(torch.clamp(var, min=1e-6))
            else:
                mean = torch.tensor(0.0)
                std = torch.tensor(1.0)

            self.domain_means[d] = mean.float()
            self.domain_stds[d] = std.float()

        self._fitted = True

    def __call__(self, img: Image.Image, domain_idx: int) -> torch.Tensor:
        """
        Apply domain harmonization + resizing + 3-channel + normalization.
        img: PIL Image (grayscale)
        domain_idx: integer domain index as in dataset.domain2idx
        """
        if not self._fitted:
            raise RuntimeError("DomainHarmonizer must be fitted before use.")

        x = self._to_tensor(img)  # shape: [1, H, W], in [0, 1]
        mu = self.domain_means[domain_idx]
        std = self.domain_stds[domain_idx]

        mu_ref = self.domain_means[self.reference_domain_idx]
        std_ref = self.domain_stds[self.reference_domain_idx]

        x = (x - mu) / std * std_ref + mu_ref
        x = x.clamp(0.0, 1.0)

        # Resize
        x = x.unsqueeze(0)  # [1, 1, H, W]
        x = torch.nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)  # [1, H', W']

        # Make 3-channel and normalize
        x3 = x.repeat(3, 1, 1)  # [3, H', W']
        x3 = (x3 - self.normalize_mean) / self.normalize_std
        return x3
