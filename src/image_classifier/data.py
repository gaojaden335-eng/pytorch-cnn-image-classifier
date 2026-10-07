from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import CLASS_NAMES


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    """Read images from one directory per class without requiring torchvision."""

    def __init__(
        self,
        split_root: Path,
        class_names: tuple[str, ...] = CLASS_NAMES,
        image_size: int = 250,
        cache_images: bool = True,
        augment: bool = False,
    ):
        self.split_root = Path(split_root)
        self.class_names = tuple(class_names)
        self.image_size = image_size
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []

        for label, class_name in enumerate(self.class_names):
            class_dir = self.split_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            paths = sorted(
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not paths:
                raise ValueError(f"No supported images found in {class_dir}")
            self.samples.extend((path, label) for path in paths)

        self.cached_images = (
            [self._read_image(path) for path, _ in self.samples]
            if cache_images else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            return np.asarray(image, dtype=np.uint8).copy()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        array = (
            self.cached_images[index]
            if self.cached_images is not None
            else self._read_image(path)
        )
        if self.augment and np.random.rand() < 0.5:
            array = np.flip(array, axis=1).copy()
        tensor = torch.from_numpy(array.astype(np.float32) / 255.0).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5, label


def create_datasets(
    data_root: Path,
    image_size: int = 250,
    cache_images: bool = True,
    augment: bool = False,
) -> tuple[Dataset, Dataset]:
    data_root = Path(data_root)
    train = ImageFolderDataset(
        data_root / "train", image_size=image_size,
        cache_images=cache_images, augment=augment,
    )
    test = ImageFolderDataset(
        data_root / "test", image_size=image_size,
        cache_images=cache_images, augment=False,
    )
    return train, test
