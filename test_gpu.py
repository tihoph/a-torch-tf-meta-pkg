"""Test GPU availability."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from a_torch_tf_meta_pkg import import_torch_before_tf

if TYPE_CHECKING:
    from collections.abc import Callable


import_torch_before_tf()


def test_fit() -> None:
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
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


def get_torch_gpu_count() -> int:
    """Get the number of available GPUs for torch."""
    import torch

    if sys.platform == "linux":  # type: ignore[unreachable,unused-ignore]
        return torch.cuda.device_count()

    if sys.platform == "darwin":  # type: ignore[unreachable,unused-ignore]
        return int(torch.backends.mps.is_available())

    raise AssertionError(f"Unsupported platform: {sys.platform}")


def get_tf_gpu_count() -> int:
    """Get the number of available GPUs for tensorflow."""
    import tensorflow as tf

    if sys.platform not in {"linux", "darwin"}:
        raise AssertionError(f"Unsupported platform: {sys.platform}")

    return len(tf.config.experimental.list_physical_devices("GPU"))


@pytest.mark.parametrize("gpu_count", [get_torch_gpu_count, get_tf_gpu_count])
def test_gpu(gpu_count: Callable[[], int]) -> None:
    """Test GPU availability."""
    assert gpu_count() > 0, f"No GPU found for {gpu_count.__name__}"


@pytest.mark.skipif(sys.platform != "Linux", reason="TensorRT only on Linux")
def test_tensorrt() -> None:
    import tensorflow as tf
    from tensorflow.python.compiler.tensorrt.trt_convert import TrtGraphConverterV2

    assert isinstance(tf.experimental.tensorrt.Converter(), TrtGraphConverterV2)
