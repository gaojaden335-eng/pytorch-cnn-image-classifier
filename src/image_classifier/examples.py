import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CLASS_NAMES
from .data import create_datasets
from .network import ImageClassifierCNN


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    return np.clip(tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5, 0.0, 1.0)


def gather(data_root: Path, checkpoint: Path, device: torch.device):
    _, test_set = create_datasets(data_root)
    loader = DataLoader(test_set, batch_size=16, shuffle=False)
    model = ImageClassifierCNN().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    images, predictions, true_labels = [], [], []
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            predictions.extend(model(batch_images.to(device)).argmax(1).cpu().tolist())
            true_labels.extend(batch_labels.tolist())
            images.extend(denormalize(image) for image in batch_images)

    by_class = {name: {"correct": [], "wrong": []} for name in CLASS_NAMES}
    for index, (predicted, true) in enumerate(zip(predictions, true_labels)):
        bucket = "correct" if predicted == true else "wrong"
        by_class[CLASS_NAMES[true]][bucket].append((index, predicted))
    return images, by_class


def render(images, by_class, output_dir: Path, per_class: int = 3) -> Path:
    figure, axes = plt.subplots(
        len(CLASS_NAMES), 2 * per_class,
        figsize=(2 * per_class * 2.0, len(CLASS_NAMES) * 2.2),
    )
    figure.suptitle("Test predictions (left: correct, right: incorrect)", fontsize=13)
    for row, class_name in enumerate(CLASS_NAMES):
        correct = by_class[class_name]["correct"][:per_class]
        wrong = by_class[class_name]["wrong"][:per_class]
        items = correct + wrong
        for col, (index, predicted) in enumerate(items):
            axis = axes[row, col]
            axis.imshow(images[index])
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{class_name} -> {CLASS_NAMES[predicted]}", fontsize=8)
        for col in range(len(items), 2 * per_class):
            axes[row, col].axis("off")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "examples_correct_wrong.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot correct and incorrect predictions")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, by_class = gather(args.data_root, args.checkpoint, device)
    print(f"Saved example grid to {render(images, by_class, args.output_dir)}")


if __name__ == "__main__":
    main()
