import torch

from image_classifier import ImageClassifierCNN


def test_model_output_shape():
    model = ImageClassifierCNN(num_classes=5)
    output = model(torch.zeros(2, 3, 250, 250))
    assert output.shape == (2, 5)


def test_model_output_is_finite():
    model = ImageClassifierCNN(num_classes=5)
    output = model(torch.randn(1, 3, 250, 250))
    assert torch.isfinite(output).all()
