# ACMP — lightweight (CPU, no-AI) image that serves the REST API or the demo.
# The heavy AI extras (torch/diffusers/ultralytics) are intentionally excluded
# to keep the image small; the pipeline runs in Ken-Burns/fallback mode here.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: glib for opencv-headless, ffmpeg for video encoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only what's needed to install + run (see .dockerignore).
COPY pyproject.toml README.md ./
COPY acmp ./acmp
COPY configs ./configs
COPY streamlit_app.py ./

RUN pip install --upgrade pip && pip install ".[api,demo]"

# 8000 = REST API (FastAPI), 8501 = Streamlit demo.
EXPOSE 8000 8501

# Default: REST API. To run the web demo instead:
#   docker run -p 8501:8501 acmp \
#     streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
CMD ["uvicorn", "acmp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
