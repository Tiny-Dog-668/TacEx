# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Use Isaac Lab's official Docker image as base
FROM nvcr.io/nvidia/isaac-sim:2023.1.1

# Set working directory
WORKDIR /workspace

# Install additional dependencies
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    opencv-python \
    pillow \
    tqdm \
    wandb

# Copy the project files
COPY . /workspace/

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV ISAAC_PATH=/isaac-sim

# Make scripts executable
RUN chmod +x /workspace/scripts/cylinder_grasping/*.py

# Set the default command
CMD ["/bin/bash"]
