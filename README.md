# a-torch-tf-meta-pkg
This package provides compatible versions of **Torch**, **TensorFlow**, **CUDA** (only on Linux),
and **TensorRT** (only on Linux) for Python 3.10 and 3.11, supporting both **macOS** and **Linux**.

| **Platform** | **Python Version** | **Torch Version** | **TensorFlow Version** | **CUDA Version** | **cuDNN Version** | **TensorRT Version** |
|--------------|--------------------|-------------------|------------------------|------------------|-------------------|----------------------|
| macOS        | 3.10               | 2.0.1             | 2.13.1                 | N/A              | N/A               | N/A                  |
| macOS        | 3.11               | 2.0.1             | 2.13.1                 | N/A              | N/A               | N/A                  |
| Linux        | 3.10               | 2.0.1             | 2.11.1                 | 11.7.99          | 8.5.0.96          | 7.2.2.3              |
| Linux        | 3.11               | 2.0.1             | 2.13.1                 | 11.7.99          | 8.6.0.163         | 8.4.3.1              |

### Installation

To install the package with development dependencies, use the following command:

```bash
git clone https://github.com/audivir/a-torch-tf-meta-pkg
cd a-torch-tf-meta-pkg
pip install ".[dev]"
```

Alternatively, you can install directly from GitHub with:

```bash
pip install git+https://github.com/audivir/a-torch-tf-meta-pkg
curl -sL https://raw.githubusercontent.com/audivir/a-torch-tf-meta-pkg/main/postinstall.sh | bash
```

### Post-Installation

Run the post-installation script to set up the python environment:

```bash
./postinstall.sh          # Run the post-installation script
# or
./postinstall.sh --mute   # Run the script without NUMA errors
```

### Testing GPU Support

Once the installation is complete, you can test GPU support by running:

```bash
if [[ $(uname) == "Linux" ]]; then
    # Set the appropriate environment variables for Linux
    LD_LIBRARY_PATH=$(python -m a_torch_tf_meta_pkg ld) \
    XLA_FLAGS=$(python -m a_torch_tf_meta_pkg xla) \
    pytest ./test_gpu.py -v
else
    # For macOS, just run the tests without extra environment variables
    pytest ./test_gpu.py -v
fi
```

### Caveats

- **Torch** should be imported **before** **TensorFlow** in the same Python process. This is necessary to avoid compatibility issues between the two frameworks.

To ensure compatibility, you can use the following function:

```python
a_torch_tf_meta_pkg.import_torch_before_tf()
```
