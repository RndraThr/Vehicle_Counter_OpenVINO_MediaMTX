FROM openvino/ubuntu20_dev:2023.0.0

USER root
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY input/ ./input/

RUN mkdir -p app/model

RUN cd app/model && \
    if [ ! -f "vehicle-detection-adas-0002.xml" ]; then \
        echo "Downloading vehicle detection model..." && \
        wget -q https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.xml && \
        wget -q https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/vehicle-detection-adas-0002/FP32/vehicle-detection-adas-0002.bin; \
    fi

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python3", "app/main.py"]