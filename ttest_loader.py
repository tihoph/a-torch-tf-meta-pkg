"""Tests a single combination."""

# ruff: noqa: T201
from __future__ import annotations

import sys


def test_compatibility() -> int:
    """Test if TF and PyTorch work."""
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.compiler.tensorrt.trt_convert import TrtGraphConverterV2

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("No TF GPUs found.")
        return 1

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        print("Failed to set memory growth")

    try:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(28, 28)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

        x_train = np.random.random((32, 28, 28))
        y_train = np.random.randint(10, size=(32,))
        # (x_test, y_test) = np.random.random((100, 28, 28)), np.random.random((100, 10)) # noqa: ERA001, E501

        model.compile(loss="sparse_categorical_crossentropy")
        model.fit(x_train, y_train, epochs=2)

    except Exception:  # noqa: BLE001
        print("TF fitting not possible.")
        return 1

    try:
        import torch

        n_torch_gpus = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        n_torch_gpus = 0

    if not n_torch_gpus:
        print("No PyTorch GPUs found")
        return 1

    if sys.argv[1] == "-1":
        print("No TensorRT installed")
        return 0

    try:
        if isinstance(tf.experimental.tensorrt.Converter(), TrtGraphConverterV2):
            return 0
    except RuntimeError:
        print("No TensorRT found")
        return 1

    raise RuntimeError("Unexpected EOF")


if __name__ == "__main__":
    raise SystemExit(test_compatibility())
