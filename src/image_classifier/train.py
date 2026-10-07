import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .data import create_datasets
from .network import ImageClassifierCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss = correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=use_amp)
        labels = labels.to(device, non_blocking=use_amp)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train(config: Config, data_root: Path, output_dir: Path) -> Path:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, _ = create_datasets(
        data_root, config.image_size, config.cache_images, config.augment
    )
    loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    model = ImageClassifierCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    losses = []
    for epoch in range(config.epochs):
        loss, accuracy = train_one_epoch(model, loader, loss_fn, optimizer, device)
        losses.append(loss)
        print(f"Epoch {epoch + 1:02d}/{config.epochs}: loss={loss:.4f}, accuracy={accuracy:.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cnn_lr_{config.learning_rate:g}"
    if config.tag:
        stem += f"_{config.tag}"
    checkpoint = output_dir / f"{stem}.pt"
    config_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }
    torch.save({"model_state": model.state_dict(), "config": config_dict}, checkpoint)

    figure, axis = plt.subplots()
    axis.plot(range(1, config.epochs + 1), losses, marker="o")
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training loss")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / f"loss_{stem}.png", dpi=160)
    plt.close(figure)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reference image-classification CNN")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--augment", action="store_true", help="enable horizontal-flip augmentation")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()
    config = Config(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        augment=args.augment,
        weight_decay=args.weight_decay,
        tag=args.tag,
        output_dir=args.output_dir,
    )
    print(f"Saved checkpoint to {train(config, args.data_root, args.output_dir)}")


if __name__ == "__main__":
    main()
