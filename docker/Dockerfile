FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
LABEL maintainer="jmahn <jmahn@ciel.co.kr>"

ARG PT_USER="ciel"
ARG PT_UID="1000"

# Install system dependencies
RUN apt-get update

# 패키지 관리자를 사용하여 GDAL 설치
#RUN apt-get install -y --fix-missing libgdal-dev

# install the base toolchain
RUN apt-get install --yes --no-install-recommends python3 python3-pip locales sudo wget build-essential python3-dev software-properties-common ca-certificates && rm -rf /var/lib/apt/lists/*

# Install python3-venv
RUN apt-get update && apt-get install -y python3-venv

# Add a symbolic link for python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip inside the virtual environment
RUN pip install --upgrade pip

# Install Base PyTorch System - Assume CPU
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies: numpy, pandas, scipy
RUN pip3 install numpy pandas spicy tables h5py

RUN pip3 install geopandas overpy shapely pandocfilters pyproj zstandard setuptools tqdm matplotlib
# for fast api 
# RUN pip3 install "fastapi[standard]” orjson

# Create a non-root user
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && useradd -l -m -s /bin/bash -N -u "${PT_UID}" "${PT_USER}" && chmod g+w /etc/passwd

WORKDIR /STDN
