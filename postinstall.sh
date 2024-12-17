#!/bin/bash

# https://stackoverflow.com/questions/44232898/memoryerror-in-tensorflow-and-successful-numa-node-read-from-sysfs-had-negativ
mute_numa_errors() {
  muted=0
  n_devices=0

  # Iterate over each device and mute NUMA errors
  find /sys/bus/pci/devices -type l | while read -r device; do
    if ! grep -q "^0$" "$device/numa_node"; then
      # Mute NUMA errors by setting numa_node to 0
      echo 0 | sudo tee "$device/numa_node" > /dev/null
      ((muted++))
    fi
    ((n_devices++))
  done

  echo "Muted $muted/$n_devices devices."
}

check_flag() {
  local flag=$1
  local short_flag=$2
  shift 2  # Shift by 2 to remove the first two arguments (flag and short_flag)

  for arg in "$@"; do
      if [[ "$arg" == "--$flag" || "$arg" == "-$short_flag" ]]; then
          return 0  # Return 0 (true) if flag is found
      fi
  done

  return 1  # Return 1 (false) if flag is not found
}

select_by_py_version() {
  # Extract the major and minor version of Python
  PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)

  # Arguments
  ten=$1
  eleven=$2

  # Check if at least one value is provided
  if [[ -z "$ten" ]]; then
    echo "Error: Please provide at least one value."
    exit 1
  fi

  # Set default for eleven if not provided
  eleven=${eleven:-$ten}

  # Check for Python version and select the appropriate value
  if [[ -z "$PYTHON_VERSION" ]]; then
    echo "Error: No Python version found."
    exit 1
  elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
    echo "$ten"
  elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
    echo "$eleven"
  else
    echo "Error: Python version $PYTHON_VERSION is not supported. Please use Python 3.10 or 3.11."
    exit 1
  fi
}


install_tensorrt() {
  # Get the correct version based on Python version
  version=$(select_by_py_version "7.2.2.3" "8.4.3.1")

  # Get the installed TensorRT version from pip
  TENSORRT_VERSION=$(pip list | grep -i "nvidia-tensorrt" | awk '{print $2}')

  # If the required version is already installed, return
  if [[ "$TENSORRT_VERSION" == "$version" ]]; then
    echo "TensorRT version $TENSORRT_VERSION is already installed. Requirement satisfied."
    return 0
  fi

  # Install the required dependencies for TensorRT version 7.2.2.3
  if [[ "$version" == "7.2.2.3" ]]; then
    echo "Installing TensorRT version $version"
    pip install \
      "nvidia-cuda-nvrtc >= 11.1, < 11.2" \
      --extra-index-url=https://pypi.nvidia.com \
      --no-deps \
      | grep -v "Looking in indexes"
  fi

  # Function to generate the wheel file name
  wheel_file() { echo "nvidia_tensorrt-$version-$1-none-linux_x86_64.whl"; }

  # Download the appropriate wheel file based on Python version
  SOURCE_FILE=$(wheel_file cp38)
  TARGET_FILE=$(wheel_file py3)

  # Download and install the TensorRT wheel file
  curl -sL "https://pypi.nvidia.com/nvidia-tensorrt/$SOURCE_FILE" -o "$TARGET_FILE" \
    && pip install "$TARGET_FILE" --no-deps \
    && rm "$TARGET_FILE"

  echo "TensorRT version $version installed successfully."
}


overwrite_cudnn() {
  pip install nvidia-cudnn-cu11==8.6.0.163 --no-deps
}

# Check if the system is Linux or Darwin
OS=$(uname)
if [[ "$OS" == "Darwin" ]]; then
    echo "This script is not necessary on Darwin."
    exit 1
elif [[ "$OS" != "Linux" ]]; then
    echo "Unsupported OS $OS."
    exit 1
fi

# Check if the version is 3.10 or 3.11
select_by_py_version 'Python version $PYTHON_VERSION is supported.'

# Run mute_numa_errors if --mute or -m is provided
if check_flag mute m $@; then
  mute_numa_errors
fi

# Run TensorRT installation
install_tensorrt

# Overwrite cuDNN if Python version is 3.11
if [[ "$PYTHON_VERSION" == "3.11" ]]; then
  echo "Python version $PYTHON_VERSION detected. Overwriting cuDNN..."
  overwrite_cudnn
else
  echo "CuDNN does not need to be overwritten for Python version $PYTHON_VERSION."
fi
