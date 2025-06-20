FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy source
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
