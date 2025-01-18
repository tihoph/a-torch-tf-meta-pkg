# %%
"""Install TensorFlow, PyTorch, and CUDA."""

# ruff: noqa: E501
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import sysconfig
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pip._vendor.packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Sequence


def get_site_packages(executable: str | Path) -> Path:
    """Get site packages of `executable`."""
    site_packages = (
        sysconfig.get_paths()["purelib"]
        if str(executable) == sys.executable
        else subprocess.check_output(  # noqa: S603
            [
                str(executable),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            text=True,
        )
    )
    return Path(site_packages)


def pip_wrapper(
    executable: str | Path,
    reqs: Sequence[str | Path],
    deps: bool = True,
    index: str | None = None,
    extra_args: Sequence[str] | None = None,
    mode: Literal["install", "download"] = "install",
    quiet: bool = True,
) -> None:
    """Install or download `reqs` with `executable`'s pip."""
    subprocess.check_call(  # noqa: S603
        [
            str(executable),
            "-m",
            "pip",
            mode,
            *(str(x) for x in reqs),
            *(() if deps else ("--no-deps",)),
            *(("--index-url", index) if index else ()),
            *(extra_args or ()),
            *(("--quiet",) if quiet else ()),
        ]
    )


def install_tensorrt(executable: str | Path, version: Version) -> None:
    """Install TensorRT."""
    hash_str = hashlib.sha256(b"a-torch-tf-meta-pkg").hexdigest()
    cache_dir = Path(f"/tmp/a-torch-tf-meta-pkg-{hash_str}")  # noqa: S108
    cache_dir.mkdir(exist_ok=True)
    if version == Version("8.6.1"):
        target = (
            cache_dir / "tensorrt_libs-8.6.1-py2.py3-none-manylinux_2_17_x86_64.whl"
        )
        if not target.exists():
            # download if not exists
            pip_wrapper(
                executable,
                [f"tensorrt-libs=={version}"],
                deps=False,
                index="https://pypi.nvidia.com",
                mode="download",
            )
        # install tensorrt (no deps needed)
        pip_wrapper(executable, [target], deps=False)
        # create missing symlinks
        site_packages = get_site_packages(executable)
        for path in site_packages.glob("tensorrt_libs/*.so.8"):
            (path.parent / f"{path.name}.6.1").symlink_to(path.name)

    elif version in {Version("7.2.2.3"), Version("8.4.3.1")}:
        target = (
            cache_dir
            / f"nvidia_tensorrt-{version}-py2.py3-none-manylinux_2_17_x86_64.whl"
        )
        if not target.exists():
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                pip_wrapper(
                    executable,
                    [f"nvidia-tensorrt=={version}"],
                    deps=False,
                    index="https://pypi.nvidia.com",
                    extra_args=("-d", tmp, "--python-version", "3.8"),
                    mode="download",
                )
                whl_path = next(tmp_path.iterdir())
                whl_path.rename(target)

        reqs = [str(target)]
        if version == Version("7.2.2.3"):
            reqs.append("nvidia-cuda-nvrtc~=11.1.0")
        pip_wrapper(executable, reqs, deps=False, index="https://pypi.nvidia.com")
    else:
        raise ValueError("Wrong TensorRT version")


@dataclass
class MetaDef:
    """Defines the compatibility settings for the meta package."""

    py: Version
    tf: Version
    torch: Version
    tensorrt: Version | None = None
    separate_torch: bool = False
    cuda_11: bool = False
    cudnn_9: bool = False
    exports: list[Literal["nvidia", "tensorrt", "tensorrt_libs"]] = field(
        default_factory=list
    )
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        tensorrt = self._tensorrt(self.tf)
        if tensorrt == -1:
            self.errors.append("TensorRT version not defined for TensorFlow version.")
        else:
            self.tensorrt = tensorrt  # type:ignore[assignment]

        if (
            not Version("3.9") <= self.py < Version("3.13")
            or not Version("2.10") <= self.tf < Version("2.19")
            or not Version("2.0") <= self.torch < Version("2.6")
        ):
            self.errors.append("Python, TensorFlow, PyTorch combination not defined.")
            return
        if (self.py >= Version("3.12") and self.tf < Version("2.16")) or (
            self.py >= Version("3.11") and self.tf < Version("2.12")
        ):
            self.errors.append("TensorFlow version too old for python => incompatible.")
        if self.tf >= Version("2.18"):
            self.warnings.append("No TensorRT support.")
        if self.torch < Version("2.1"):
            if self.tf >= Version("2.18"):
                self.errors.append(
                    "TensorFlow needs CUDA12, torch CUDA11 => incompatible."
                )
            elif self.tf >= Version("2.12"):
                self.errors.append(
                    "TensorFlow needs cuDNN 8.6 or higher, torch cuDNN 8.6 => incompatible."
                )
            else:
                self.exports.extend(("nvidia", "tensorrt"))  # needs to be in env
        elif self.tf >= Version("2.18"):
            self.cudnn_9 = True
        else:
            if self.torch >= Version("2.4") or self.tf < Version("2.15"):
                self.cuda_11 = True
            if self.tf >= Version("2.15"):
                self.exports.append("tensorrt_libs")
            elif self.tf >= Version("2.14"):
                pass  # all fine
            else:  # tf < Version("2.14")
                self.exports.extend(("nvidia", "tensorrt"))
            if Version("2.13") <= self.tf < Version("2.14") and self.torch >= Version(
                "2.2"
            ):
                self.warnings.append("PyTorch will be installed separately.")
                self.separate_torch = True

        if self.cuda_11 and self.cudnn_9:
            self.errors.append("CUDA 11 and cuDNN 9 not supported.")

    @classmethod
    def _tensorrt(cls, tf: Version) -> Version | None | Literal[-1]:
        """TensorRT version for TensorFlow version."""
        if tf < Version("2.12"):
            return Version("7.2.2.3")
        if tf < Version("2.15"):
            return Version("8.4.3.1")
        if tf < Version("2.18"):
            return Version("8.6.1")
        if Version("2.18") <= tf < Version("2.19"):
            return None
        return -1

    @property
    def install_requires(self) -> list[str]:
        """Install requirements."""
        reqs = [
            "numpy<2.0",
            f"tensorflow~={self.tf}",
            'tensorflow-metal; platform_system == "Darwin"',
        ]
        if not self.separate_torch:
            reqs.append(f"torch=={self.torch}")
        if self.cuda_11:
            reqs.extend(
                [
                    'nvidia-cuda-runtime-cu11==11.8.89; platform_system == "Linux" and platform_machine == "x86_64"',
                    'nvidia-cudnn-cu11==8.9.6.50; platform_system == "Linux" and platform_machine == "x86_64"',
                    'nvidia-cublas-cu11; platform_system == "Linux" and platform_machine == "x86_64"',
                    'nvidia-cusparse-cu11; platform_system == "Linux" and platform_machine == "x86_64"',
                    'nvidia-cufft-cu11; platform_system == "Linux" and platform_machine == "x86_64"',
                ]
            )
        elif self.cudnn_9:
            reqs.append(
                'nvidia-cudnn-cu11==9.3.0.75; platform_system == "Linux" and platform_machine == "x86_64"'
            )
        return reqs

    @property
    def python_requires(self) -> str:
        """Python requirements."""
        return ">=3.9,<3.13"

    @property
    def platforms(self) -> list[str]:
        """Platform requirements."""
        if meta_def.tf < Version("2.13"):
            return ["Linux"]
        return ["Darwin", "Linux"]


if __name__ == "__main__":
    from setuptools import setup
    from setuptools.command.install import install

    meta_pkg = "a_torch_tf_meta_pkg"
    tf = Version("2.15.0")
    torch = Version("2.5.1")
    py = Version(".".join(platform.python_version_tuple()[:2]))

    meta_def = MetaDef(py, tf, torch)
    short_tf = "".join(str(x) for x in meta_def.tf.release[:2])
    short_torch = "".join(str(x) for x in meta_def.torch.release)
    version = f"{short_tf}.{short_torch}"

    if meta_def.errors:
        raise ValueError(" ".join(meta_def.errors))

    if meta_def.warnings:
        warnings.warn(" ".join(meta_def.warnings), RuntimeWarning, stacklevel=1)

    class InstallCommand(install):
        """Installs the meta package."""

        def run(self) -> None:
            """Run the installation process."""
            if meta_def.separate_torch:
                pip_wrapper(sys.executable, [f"torch=={torch}"])

            if (
                meta_def.tensorrt
                and platform.system() == "Linux"
                and platform.machine() == "x86_64"
            ):
                install_tensorrt(sys.executable, meta_def.tensorrt)

            super().run()

    setup(
        name=meta_pkg,
        version=version,
        author="Tim Hörmann",
        author_email="pypi@audivir.de",
        classifiers=[
            "License :: Other/Proprietary License",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
        ],
        py_modules=[meta_pkg],
        python_requires=meta_def.python_requires,
        install_requires=meta_def.install_requires,
        platforms=meta_def.platforms,
        zip_safe=True,
        keywords="tensorflow pytorch cudnn nvidia tensorrt deeplearning",
        url="https://github.com/tihoph/a-torch-tf-meta-pkg",
        download_url="https://github.com/tihoph/a-torch-tf-meta-pkg/tags",
        cmdclass={"install": InstallCommand},
    )