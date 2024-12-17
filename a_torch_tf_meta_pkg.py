"""Helper module to fix the LD_LIBRARY_PATH for nvidia libraries.

As tensorflow 2.11.1 is not able to find the nvidia libraries
in the site-packages, this module is used to append the nvidia

Usage:
    ```bash
    LD_LIBRARY_PATH=$(python -m a_torch_tf_meta_pkg ld) \
    XLA_FLAGS=$(python -m a_torch_tf_meta_pkg xla) \
    python your_script.py
    ```
"""

from __future__ import annotations

import argparse
import os
import sys
import sysconfig
from pathlib import Path


def get_nvidia_ld_lib_path() -> str:
    """Get the LD_LIBRARY_PATH with nvidia libraries appended.

    Returns:
        The new LD_LIBRARY_PATH with nvidia libraries appended.

    Raises:
        FileNotFoundError: If no nvidia libraries are found in site-packages.
    """
    prev_ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
    site_packages = Path(sysconfig.get_paths()["purelib"])
    nvidia_libs = list(site_packages.glob("nvidia/*/lib"))

    tensorrt_lib = site_packages / "tensorrt"
    if tensorrt_lib.exists():
        nvidia_libs.append(tensorrt_lib)

    if not nvidia_libs:
        raise FileNotFoundError("No nvidia libraries found in site-packages.")

    ld_lib_path = prev_ld_lib_path

    for lib in nvidia_libs:
        lib_str = str(lib)
        if lib_str not in ld_lib_path:
            ld_lib_path += f":{lib_str}"

    return ld_lib_path


def get_nvidia_xla_flags() -> str:
    """Get the XLA_FLAGS with xla_gpu_cuda_data_dir set to nvidia libraries.

    Returns:
        The new XLA_FLAGS with xla_gpu_cuda_data_dir set.

    Raises:
        FileNotFoundError: If no nvidia libraries are found in site-packages.
        ValueError: If multiple nvidia libraries are found in site-packages.
    """
    site_packages = sysconfig.get_paths()["purelib"]
    nvvm_dirs = list(Path(site_packages).glob("nvidia/**/nvvm"))

    if not nvvm_dirs:
        raise FileNotFoundError("No nvvm directories found in site-packages.")

    if len(nvvm_dirs) > 1:
        raise ValueError("Multiple nvvm directories found in site-packages.")

    xla_cuda_data_dir = nvvm_dirs[0].parent
    return f"--xla_gpu_cuda_data_dir={xla_cuda_data_dir}"


def import_torch_before_tf() -> None:
    """Import torch before tensorflow to avoid CUDA issues."""
    import torch

    del torch
    import tensorflow as tf

    del tf


def _main() -> int:
    """Print the flags."""
    parser = argparse.ArgumentParser(
        description="Fix the LD_LIBRARY_PATH and XLA_FLAGS environment variables for nvidia libraries"  # noqa: E501
    )
    parser.add_argument(
        "cmd",
        choices=("ld", "xla"),
        help="Print either the LD_LIBRARY_PATH or the XLA_FLAGS with nvidia libraries appended.",  # noqa: E501
    )
    args = parser.parse_args()
    if args.cmd == "ld":
        print(get_nvidia_ld_lib_path())  # noqa: T201
    elif args.cmd == "xla":
        print(get_nvidia_xla_flags())  # noqa: T201

    return 0


__all__ = ["get_nvidia_ld_lib_path", "get_nvidia_xla_flags", "import_torch_before_tf"]

if __name__ == "__main__":
    if sys.platform not in {"linux", "darwin"}:
        raise AssertionError(f"Unsupported platform: {sys.platform}")

    if sys.platform == "darwin":
        print("No environment update needed on macOS.")  # noqa: T201
        raise SystemExit(1)

    raise SystemExit(_main())
