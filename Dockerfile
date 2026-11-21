# Container deploy (Railway / Render / ECS / App Runner).
# Do NOT use AWS Lambda zip — Torch alone exceeds the 500 MB function limit.

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY material_intelligence ./material_intelligence
COPY app.py main.py ./
COPY asset ./asset
COPY .env.example ./.env.example

RUN pip install --upgrade pip \
    && pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install -e . --no-deps

RUN mkdir -p data/uploads data/models data/datasets data/reports

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
