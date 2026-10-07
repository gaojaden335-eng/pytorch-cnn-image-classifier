import argparse
from pathlib import Path

import torch
from torch import nn

from .config import CLASS_NAMES
from .data import create_datasets
from .network import ImageClassifierCNN


def validate(data_root: Path) -> None:
    train_set, test_set = create_datasets(data_root)
    image, label = train_set[0]
    assert image.shape == (3, 250, 250)
    assert image.dtype == torch.float32
    assert 0 <= label < len(CLASS_NAMES)
    assert len(train_set) > 0 and len(test_set) > 0
    model = ImageClassifierCNN(num_classes=len(CLASS_NAMES))
    logits = model(torch.zeros(2, 3, 250, 250))
    assert logits.shape == (2, len(CLASS_NAMES))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    print("Validation passed: dataset, tensor shape, labels, model output, and loss are valid.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the dataset and model")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    validate(args.data_root)


if __name__ == "__main__":
    main()
