"""Helper module to fix the LD_LIBRARY_PATH for nvidia libraries.

Usage:
    ```bash
    export LD_LIBRARY_PATH=$(python a_torch_tf_meta_pkg)
    python ...
    # or alternatively
    LD_LIBRARY_PATH=$(python a_torch_tf_meta_pkg) python ...
    ```
"""

# ruff: noqa: INP001, E501
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


def get_nvidia_env() -> int:
    """Prints the fixed LD_LIBRARY_PATH for the environment."""
    site_packages = Path(sysconfig.get_paths()["purelib"])

    nvidia_libs = list(site_packages.glob("nvidia/*/lib"))
    for tensorrt_name in ("tensorrt", "tensorrt_libs"):
        if (site_packages / tensorrt_name).exists():
            nvidia_libs.append(site_packages / tensorrt_name)
            break

    nvidia_str = ":".join(str(x) for x in nvidia_libs)
    if prev := os.getenv("LD_LIBRARY_PATH"):
        nvidia_str += f":{prev}"
    print(nvidia_str)  # noqa: T201

    return 0


def import_torch_before_tf() -> None:
    """Import torch before tensorflow to avoid CUDA issues."""
    import torch

    del torch
    import tensorflow as tf

    del tf


__all__ = ["import_torch_before_tf"]

if __name__ == "__main__":
    if sys.platform not in {"linux", "darwin"}:
        raise AssertionError(f"Unsupported platform: {sys.platform}")

    if sys.platform == "darwin":
        print("No environment update needed on macOS.", file=sys.stderr)  # noqa: T201
        raise SystemExit(0)

    raise SystemExit(get_nvidia_env())
