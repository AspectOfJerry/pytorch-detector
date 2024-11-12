# PyTorch object detection setup

### Requirements

| Name   | Version |
|--------|---------|
| Python | 3.9     |

### Dataset structure

The dataset should be structured as follows:

```
dataset
├── temp (rejected images, should not happen)
├── annotations
│   ├── test
│   └── train
└── images
    ├── test
    └── train
```

### Run tensorboard (optional)

TensorBoard can be used to monitor the training process, including losses and metrics. To start TensorBoard, run:

```bash
tensorboard --logdir=./output/tensorboard_logs
```

This will display training progress in your browser.

### Using a venv (Recommended)

Creating a virtual environment helps manage project dependencies independently of other Python projects.
If you use PyCharm, its virtual environment manager makes this very easy and straightforward.

Alternatively, to set it up manually:

```bash
python -m venv venv
```

Activate the virtual environment before proceeding with the installation of dependencies. You might need to reopen a new terminal.

## Installing dependencies

From [PyTorch's website](https://pytorch.org/get-started/locally/), here are the installation commands:

### CPU-only

Use this command to install PyTorch with only CPU support:

```bash
pip install torch torchvision torchaudio
```

### GPU (NVIDIA CUDA)

Use this command to install PyTorch with GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Common dependencies

And install the other necessary libraries:

```bash
pip install -r requirements.txt
```

```
torchinfo~=1.8.0
opencv-python~=4.10.0.84
pillow~=10.3.0
tensorboard
```

> **Note**: PyTorch’s GPU version comes bundled with all necessary binaries, such as CUDA and cuDNN, so you don’t need to install them separately.

<br>

~~### Installing CUDA and cuDNN~~

~~If you have an NVIDIA GPU, you can install the CUDA toolkit and cuDNN to enable GPU support. If you don't have an NVIDIA GPU, skip the following steps.~~

~~- Download and install [CUDA 12.4 for Windows](https://developer.nvidia.com/cuda-12-4-0-download-archive)~~
~~- Download and install [Visual Studio](https://visualstudio.microsoft.com/) with C++ build tools~~
~~- In the Visual Studio Installer, under Workloads, select Desktop development with C++~~
~~- **or** Under Individual components, select MSVC v... - VS 20.. C++ x64/x86 build tools *(not sure if that works. i selected the full c++ bundle)*~~
~~- Proceed with the installation~~
~~- Download [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) (any compatible version)~~
~~- Create an NVIDIA developer account if you don't have one~~
~~- Extract the cuDNN zip file~~
~~- Copy the files from the `bin` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`~~
~~- Copy the files from the `include` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include`~~
~~- Copy the files from the `lib\x64` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64`~~
