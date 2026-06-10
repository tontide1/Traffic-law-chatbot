FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  libpq-dev \
  gcc \
  libgl1 \
  libglib2.0-0 \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Use uv for faster dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system -r requirements.txt

COPY . ./backend

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
