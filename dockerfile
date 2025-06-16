# Start with Python
FROM python:3.13.4
# Set up workspace
WORKDIR /app

# Copy requirements first (Docker caching magic)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy your agent code
COPY . .
# Expose port for API
EXPOSE 8000
# Start your agent
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]