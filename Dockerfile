FROM python:3.12-slim

# ffmpeg cuts and renders, libmagic is used by ClipsAI to identify files,
# DejaVu is the default caption font.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libmagic1 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU PyTorch by default (most servers have no GPU). For an NVIDIA GPU build with:
#   docker compose build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url ${TORCH_INDEX}

COPY pyproject.toml README.md constraints.txt ./
COPY clipper ./clipper
RUN pip install --no-cache-dir -c constraints.txt ".[clip]" \
 && python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab

ENV PYTHONUNBUFFERED=1 \
    CLIPPER_CONFIG=/app/config.yaml \
    HF_HOME=/app/data/models
VOLUME ["/app/data"]
ENTRYPOINT ["clipper"]
CMD ["run"]
