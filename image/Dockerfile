# Stage 1: Build dependencies
FROM python:3.11-slim as builder

# Install build tools and system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Upgrade pip and install Python packages into a temp dir
RUN python -m pip install --upgrade pip
RUN pip install --prefix=/opt/python -r requirements.txt

# Stage 2: Create minimal Lambda image
FROM public.ecr.aws/lambda/python:3.11

# Copy installed dependencies from builder stage
COPY --from=builder /opt/python /opt/python

# Set working directory and copy app source
WORKDIR ${LAMBDA_TASK_ROOT}
COPY src ${LAMBDA_TASK_ROOT}
COPY src/keys /keys


# Set environment variable (avoid using for secrets in prod)
ENV GOOGLE_APPLICATION_CREDENTIALS="/keys/curious-system-463501-g0-17b4fb32c15b.json"

EXPOSE 8000
CMD ["main.handler"]
