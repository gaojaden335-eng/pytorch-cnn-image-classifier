import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .config import CLASS_NAMES
from .data import create_datasets
from .network import ImageClassifierCNN


def evaluate(checkpoint: Path, data_root: Path, output_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_set = create_datasets(data_root)
    loader = DataLoader(test_set, batch_size=16, shuffle=False)
    model = ImageClassifierCNN().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())

    labels = range(len(CLASS_NAMES))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    print(classification_report(
        targets, predictions, labels=labels, target_names=CLASS_NAMES,
        digits=4, zero_division=0,
    ))
    print(f"Overall accuracy: {np.mean(np.array(targets) == np.array(predictions)):.4f}")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=axis, cmap="Blues", values_format="d"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved CNN checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
