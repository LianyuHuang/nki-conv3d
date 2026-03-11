FROM --platform=linux/amd64 python:3.12-slim

# Install neuronx-cc and numpy from AWS Neuron repo
RUN pip install --no-cache-dir \
    neuronx-cc==2.* \
    numpy \
    pytest \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com

# Install CPU-only PyTorch separately (different index)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY . /app

CMD ["pytest", "test_conv3d.py", "-v", "--tb=short"]
