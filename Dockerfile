FROM nvcr.io/nvidia/pytorch:21.05-py3
MAINTAINER Keiller Nogueira <keillernogueira@gmail.com>

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache --upgrade tensorboard Cython onnx gsutil matplotlib \
    opencv-python Pillow PyYAML scipy torch torchvision tqdm seaborn pandas \
    scikit-learn pycocotools thop dlib py7zr validators numpy

