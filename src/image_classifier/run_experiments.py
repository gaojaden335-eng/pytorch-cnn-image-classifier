import argparse
from pathlib import Path

from .config import Config
from .train import train


LEARNING_RATES = (0.01, 0.001, 0.0001, 0.00001, 0.000001)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one CNN per learning rate")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    for learning_rate in LEARNING_RATES:
        config = Config(
            learning_rate=learning_rate,
            epochs=args.epochs,
            output_dir=args.output_dir,
        )
        checkpoint = train(config, args.data_root, args.output_dir)
        print(f"Finished lr={learning_rate:g}: {checkpoint}")


if __name__ == "__main__":
    main()
