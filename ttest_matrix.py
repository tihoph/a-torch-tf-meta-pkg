"""Test all combinations from Python, TensorFlow, and PyTorch."""

# ruff: noqa: S603
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pip._vendor.packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Sequence

    from setup import MetaDef

TEST_FILE = Path(__file__).parent / "ttest_loader.py"


def _create_venv(py: Version, venv: str | Path) -> None:
    """Creates a venv at `venv` with version `py`."""
    conda_envs = Path(os.environ["CONDA_PREFIX"]).parent
    conda_py_bin = f"{conda_envs}/py{str(py).replace('.', '')}/bin/python"

    subprocess.check_output([str(conda_py_bin), "-m", "pip", "install", "virtualenv"])
    subprocess.check_output([str(conda_py_bin), "-m", "virtualenv", str(venv)])


def _create_env(
    venv: Path,
    py: Version,
    export: Sequence[Literal["nvidia", "tensorrt", "tensorrt_libs"]],
) -> dict[str, str]:
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "0"
    lib_paths: list[Path] = []
    if "nvidia" in export:
        lib_paths.extend(venv.glob(f"lib/python{py}/site-packages/nvidia/*/lib"))
    if tensorrt_match := {"tensorrt", "tensorrt_libs"} & set(export):
        if len(tensorrt_match) == 2:  # noqa: PLR2004
            raise ValueError("only one tensorrt export supported")
        tensorrt = tensorrt_match.pop()
        lib_paths.append(venv / f"lib/python{py}/site-packages/{tensorrt}")
    env["LD_LIBRARY_PATH"] = ":".join(str(x) for x in lib_paths)
    return env


def test_def(meta_def: MetaDef) -> None:
    """Test the meta definition."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        venv = tmp_path / "venv"
        _create_venv(meta_def.py, venv)
        # TODO: reimplements setup.py logic
        meta_def.install(venv)  # type: ignore[attr-defined]
        try:
            subprocess.check_call(
                [
                    str(venv / "bin/python"),
                    str(TEST_FILE),
                    str(meta_def.tensorrt or -1),
                ],
                env=_create_env(venv, Version("3.10"), meta_def.exports),
            )
            if meta_def.errors:
                raise ValueError(f"Errors not raised: {' '.join(meta_def.errors)}")
        except subprocess.CalledProcessError as e:
            if not meta_def.errors:
                raise ValueError(f"not expected errors: {e}") from e
