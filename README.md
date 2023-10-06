# Setup

### Requirements

| Name   | Version |
|--------|---------|
| Python | 3.9     |

### Instructions

To run the project in a Python virtual environment, run the following commands:

```bash
python -m venv venv
```

If you have an NVIDIA GPU, you can install the CUDA toolkit and cuDNN to enable GPU support. If you don't have an NVIDIA GPU, the following steps.

- Download and install [CUDA 11.2 for Windows](https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64)
- Download and install [Visual Studio 2019](https://learn.microsoft.com/en-us/visualstudio/releases/2019/redistribution#--download) with C++ build tools
    - In the Visual Studio Installer, under Workloads, select Desktop development with C++
    - **or** Under Individual components, select MSVC v142 - VS 2019 C++ x64/x86 build tools *(not sure if that works. i selected the full c++ bundle)*
    - Proceed with the installation
- Download [cuDNN v8.1.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.0.77/11.2_20210127/cudnn-11.2-windows-x64-v8.1.0.77.zip) (
  direct download) or any compatible version
    - Create an NVIDIA developer account if you don't have one

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the setup script:

```bash
python setup.py
```