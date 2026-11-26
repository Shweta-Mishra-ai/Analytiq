# ── Stage 1: build frontend ──────────────────────────────
FROM node:22-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: backend + built frontend ────────────────────
FROM python:3.11-slim
WORKDIR /srv

# ffmpeg powers local video-frame extraction for video-to-dataset
# uploads (services/video_frames.py) — without it, that endpoint fails
# with a clear "ffmpeg is not installed" error rather than a crash, but
# the feature is silently unavailable. python:3.11-slim doesn't include
# it by default.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY --from=frontend /fe/dist ./static

# REQUIRE_AUTH: a deployment that forgets APP_ADMIN_KEY is locked, not
# open to anyone with its address (see app/config.py).
ENV DATA_DIR=/srv/data \
    PORT=8000 \
    REQUIRE_AUTH=1
RUN mkdir -p /srv/data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
