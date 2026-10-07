# PyTorch Image Classification CNN

A compact PyTorch project for training and evaluating a convolutional neural network
on a five-class image dataset. It includes configurable training,
learning-rate experiments, checkpointing, confusion-matrix generation, and qualitative
prediction visualizations.

This is a source-first public release. The original dataset, face images, model weights,
local environments, and unrelated submission materials are intentionally excluded for
privacy and redistribution safety.

## Features

- Small two-block CNN with batch normalization
- Folder-based train/test datasets
- Seeded training runs for more consistent experiments
- Optional horizontal-flip augmentation and weight decay
- Batch learning-rate experiments
- Classification reports, confusion matrices, and prediction examples
- Lightweight unit tests that do not require the dataset

## Project structure

```text
.
|-- src/image_classifier/   # package source
|-- tests/                  # dataset-free unit tests
|-- docs/                   # data layout and usage notes
|-- results/                # non-sensitive aggregate result example
|-- pyproject.toml
|-- requirements.txt
`-- LICENSE
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Dataset layout

Prepare your own licensed dataset using this structure:

```text
data/
|-- train/
|   |-- airplanes/
|   |-- cars/
|   |-- dog/
|   |-- faces/
|   `-- keyboard/
`-- test/
    |-- airplanes/
    |-- cars/
    |-- dog/
    |-- faces/
    `-- keyboard/
```

Supported image formats are JPEG, PNG, BMP, and WebP. See
[`docs/data-format.md`](docs/data-format.md) for details.

The reference implementation uses the fixed class order `faces`, `dog`, `airplanes`,
`keyboard`, and `cars`. To use different classes, update `CLASS_NAMES` in
`src/image_classifier/config.py` and keep the number of model outputs consistent.

## Model architecture

The reference network accepts `3 x 250 x 250` RGB tensors and applies:

1. `7 x 7` convolution, batch normalization, ReLU, and max pooling
2. `3 x 3` convolution, batch normalization, ReLU, and max pooling
3. A fully connected layer from `64 x 30 x 30` features to five logits

Images are converted to RGB, resized to `250 x 250`, and normalized to `[-1, 1]`.

## Quick start

Validate the model and dataset:

```bash
image-classifier-validate --data-root data
```

Train one model:

```bash
image-classifier-train --data-root data --learning-rate 0.001 --epochs 20
```

Run the learning-rate sweep:

```bash
image-classifier-experiments --data-root data --epochs 20
```

Evaluate a checkpoint:

```bash
image-classifier-evaluate outputs/cnn_lr_0.001.pt --data-root data
```

Generate a grid of correct and incorrect predictions:

```bash
image-classifier-examples outputs/cnn_lr_0.001.pt --data-root data
```

All generated checkpoints and plots are written to `outputs/`, which is ignored by Git.
The commands require a dataset matching the documented folder structure; the repository
does not download or bundle one automatically.

## Example results

On the original five-class experimental dataset, a learning rate of `1e-4` reached 70%
test accuracy without augmentation and 76% with horizontal-flip augmentation. The
confusion matrix below contains aggregate counts only. Because the original dataset and
weights are not redistributed, these archived results cannot be reproduced from this
repository alone. Performance will vary with the dataset and split.

![Confusion matrix](results/confusion_matrix.png)

## Tests

```bash
pytest
```

The included tests verify model output shape and finite predictions without loading a
dataset. End-to-end training and evaluation require user-provided images.

## Privacy and security

Do not commit datasets containing personal images unless you have explicit permission and
an appropriate license. Keep secrets in environment variables, never in source files.
See [`SECURITY.md`](SECURITY.md) for reporting security issues.

## License

The source code is available under the [MIT License](LICENSE). No license is granted for
third-party datasets or images; obtain and follow their original terms separately.
