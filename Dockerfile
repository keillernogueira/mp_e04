FROM nvcr.io/nvidia/pytorch:21.05-py3
MAINTAINER Keiller Nogueira <keillernogueira@gmail.com>

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# Install python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install --no-cache --upgrade numpy scipy imageio yacs tqdm seaborn pandas scikit-learn \
	torch==1.10.2 torchvision keras tensorflow-gpu tensorboard Cython onnx gsutil matplotlib \
    opencv-python Pillow==8.2.0 PyYAML openpyxl mathfilters \
    pycocotools thop dlib py7zr validators pafy youtube_dl==2020.12.2
