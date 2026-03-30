FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Upgrade pip to avoid compatibility issues
RUN pip install --upgrade pip

# Force install PyTorch >= 2.4 and Hugging Face ecosystem
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision \
    torchaudio \
    transformers==4.40.0 \
    datasets \
    accelerate \
    torchmetrics \
    tensorboard

COPY . /app



