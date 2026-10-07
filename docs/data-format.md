# Data format

The loader expects two split directories, `train` and `test`. Each split contains one
directory per class. The directory name becomes the class label.

By default the project uses this fixed class order:

1. `faces`
2. `dog`
3. `airplanes`
4. `keyboard`
5. `cars`

Each class must exist in both splits and contain at least one supported image. Images are
converted to RGB, resized to 250 by 250 pixels, converted to tensors, and normalized to
the range `[-1, 1]`.

The dataset is deliberately not included. Use only images that you have permission to
process and redistribute. Treat face images as personal data and follow all applicable
privacy and consent requirements.
