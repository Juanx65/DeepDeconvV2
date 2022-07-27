# Use Cuda 11.5 as base
FROM nvidia/cuda:11.5.0-cudnn8-devel-ubuntu20.04 as bases


# Install basics
RUN apt-get update
RUN apt-get upgrade -y 
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get install -y git unzip 
RUN apt-get install -y python3-dev

RUN apt-get install -y python3-pip
RUN pip install --upgrade pip

# Copy folder to /opt/DeepDeconvV2

COPY ./ /opt/DeepDeconvV2


# Install Project
WORKDIR /opt/DeepDeconvV2
RUN pip install -r requirements.txt