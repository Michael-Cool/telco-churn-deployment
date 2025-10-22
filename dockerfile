# === Base Image ===
FROM python:3.11-slim

# === Arbeitsverzeichnis im Container ===
WORKDIR /app

# === Dependencies kopieren und installieren ===
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === Quellcode kopieren ===
COPY src ./src

# === Modelle separat kopieren (wichtiger Fix) ===
COPY src/models ./src/models

# === Port 8000 für FastAPI ===
EXPOSE 8000

# === Startbefehl ===
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]