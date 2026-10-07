from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ("faces", "dog", "airplanes", "keyboard", "cars")
NUM_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class Config:
    """Training defaults for the reference five-class experiment."""

    image_size: int = 250
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 1e-3
    seed: int = 338
    num_workers: int = 0
    cache_images: bool = True
    output_dir: Path = Path("outputs")
    augment: bool = False
    weight_decay: float = 0.0
    tag: str = ""
