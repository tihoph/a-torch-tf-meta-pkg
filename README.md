# a-torch-tf-meta-pkg
This package provides compatible versions of **Torch**, **TensorFlow**, **CUDA** (only on Linux),
and **TensorRT** (only on Linux), supporting both **macOS** and **Linux**.

- TensorFlow < 2.13 is only supported on Linux.
- On Python 3.11, only TensorFlow >= 2.11 is supported.
- On Python 3.12, only TensorFlow >= 2.16 is supported.
- TensorRT is downloaded from https://pypi.nvidia.com
- If PyTorch and TensorFlow require different cuDNN versions,
TensorFlow uses the nvidia-cuda-cudnn-cu11 package from PyPI,
whereas PyTorch uses nvidia-cuda-cudnn-cu12 package.
- TensorRT is not used with TensorFlow ~= 2.18.0
- PyTorch < 2.1 is only supported by TensorFlow < 2.12
- TensorFlow ~= 2.13.0 has strict typing-extensions requirements. In this case, torch is installed separately.

## Installation

To install the package with development dependencies, use the following command:

```bash
# for TensorFlow 2.14 and PyTorch 2.3.1
pip install git+https://github.com/tihoph/a-torch-tf-meta-pkg@v214.231
```

## Usage

```bash
# Sets the environment variable (LD_LIBRARY_PATH)
# not necessary on macOS
export LD_LIBRARY_PATH=$(python a_torch_tf_meta_pkg)
python ...
# or alternatively
LD_LIBRARY_PATH=$(python a_torch_tf_meta_pkg) python ...
```

## Test

```bash
pip install pytest
curl -sL https://raw.githubusercontent.com/tihoph/a-torch-tf-meta-pkg/refs/heads/main/test_gpu.py -o test_gpu.py
LD_LIBRARY_PATH=$(python a_torch_tf_meta_pkg) pytest test_gpu.py
```
