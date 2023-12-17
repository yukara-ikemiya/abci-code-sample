# Pytorch 22.04 
FROM nvcr.io/nvidia/pytorch:22.04-py3 AS ngc22.04

ARG APT_INSTALL="apt-get -y install --no-install-recommends"
ARG PIP_INSTALL="pip3 install --no-cache-dir"

# HuggingFace Accelerate
RUN ${PIP_INSTALL} accelerate