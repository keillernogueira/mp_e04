FROM nvcr.io/nvidia/pytorch:21.05-py3
MAINTAINER Keiller Nogueira <keillernogueira@gmail.com>

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache Cython coremltools onnx gsutil notebook matplotlib numpy opencv-python Pillow PyYAML scipy torch torchvision tqdm tensorboard seaborn pandas scikit-learn pycocotools thop dlib py7zr

