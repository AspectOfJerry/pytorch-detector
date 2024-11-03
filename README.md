# Setup

### Requirements

| Name   | Version |
|--------|---------|
| Python | 3.9     |

### Dataset structure

The dataset should be structured as follows:

```
dataset
├── annotations
│   ├── test
│   └── train
└── images
    ├── test
    └── train
```

### Instructions

To run the project in a Python virtual environment, run the following commands:

```bash
python -m venv venv
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU, you can install the CUDA toolkit and cuDNN to enable GPU support. If you don't have an NVIDIA GPU, skip the following steps.

- Download and install [CUDA 12.4 for Windows](https://developer.nvidia.com/cuda-12-4-0-download-archive)
- Download and install [Visual Studio](https://visualstudio.microsoft.com/) with C++ build tools
    - In the Visual Studio Installer, under Workloads, select Desktop development with C++
    - **or** Under Individual components, select MSVC v... - VS 20.. C++ x64/x86 build tools *(not sure if that works. i selected the full c++ bundle)*
    - Proceed with the installation
- Download [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) (any compatible version)
    - Create an NVIDIA developer account if you don't have one
- Extract the cuDNN zip file
    - Copy the files from the `bin` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`
    - Copy the files from the `include` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include`
    - Copy the files from the `lib\x64` folder to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64`
- Download the dependencies using:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
