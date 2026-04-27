FROM python:3.11-slim

WORKDIR /app

# Install system deps for PyMuPDF / unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Data directory for ChromaDB + SQLite (override via volume mount)
RUN mkdir -p /data/chroma

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
