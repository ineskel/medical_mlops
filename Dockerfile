FROM python:3.11-slim

WORKDIR /app

# system deps (needed for torch + image libs)
RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# expose FastAPI port
EXPOSE 8000

# run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]