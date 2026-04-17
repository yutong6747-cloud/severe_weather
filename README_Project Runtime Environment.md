
Project Runtime Environment

This repository is based on the Ultralytics YOLO framework and includes multiple custom scripts for training, validation, and data processing. The recommended environment is as follows:

Python 3.8 or higher
PyTorch 1.8 or higher
Windows, Linux, or macOS (for training, an NVIDIA GPU with CUDA is recommended)
If using Windows for training, it is recommended not to use PyTorch version 2.4.0 due to known compatibility issues

It is recommended to first create an isolated virtual environment before installing dependencies. If you only run standard detection, validation, or training scripts, the core dependencies are sufficient. Additional optional dependencies are required for certain experimental scripts.

Installation

It is recommended to execute the following steps in the root directory of the repository:

Create and activate a virtual environment
Install PyTorch that matches your CUDA version
Install the project dependencies

Example:
pip install -e .
If you only want to quickly try the official YOLO functionality, you can install Ultralytics directly:
pip install ultralytics


Optional Dependencies

The following dependencies are not required for all scripts, but are used in some experimental scripts:

albumentations
pycocotools
prettytable
timm
tidecv
mmcv
mmengine
mamba-ssm
DCNv3 / DCNv4
natten

