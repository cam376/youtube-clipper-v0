# YouTube Clipper v0.2.1 — single-container image (CPU only).
#
#   docker build -t youtube-clipper:v0.2.1 .
#   docker run -d --name clipper -p 127.0.0.1:8000:8000 -v clipper_data:/data \
#       -e OLLAMA_URL=http://host.docker.internal:11434 youtube-clipper:v0.2.1
#
# Or use docker-compose.yml, which also runs Ollama. See DEPLOYMENT.md.

FROM python:3.12-slim-trixie

# ffmpeg (Debian trixie ships 7.1 with libass) and a font for the burned
# subtitles (fontconfig maps "Arial" to Liberation Sans). The glib runtime
# that opencv-python-headless needs comes in as a dependency of ffmpeg.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-liberation \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/clipper

COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY app ./app
COPY static ./static

# Production defaults. Every one of these can be overridden at run time.
ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    OUTPUT_DIR=/data/output \
    HF_HOME=/data/models \
    WHISPER_MODEL=small \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE=int8 \
    OLLAMA_MODEL=qwen2.5:3b \
    OLLAMA_URL=http://ollama:11434 \
    MAX_CONCURRENT_JOBS=1 \
    DEBUG_FACES=false

# /data holds generated clips (OUTPUT_DIR) and the downloaded Whisper model
# cache (HF_HOME). Mount a volume there so neither is lost on restart.
RUN useradd --create-home --uid 1000 clipper \
 && mkdir -p /data/output /data/models \
 && chown -R clipper:clipper /data /srv/clipper
USER clipper
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "app/main.py"]
