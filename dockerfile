# === Base Image: Minimal Debian mit Python ===
FROM debian:bookworm-slim

# 1️⃣ System & Python installieren
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 2️⃣ Arbeitsverzeichnis anlegen
WORKDIR /app

# 3️⃣ Dependencies installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Quellcode kopieren
COPY src ./src

# 5️⃣ Port öffnen
EXPOSE 8000

# 6️⃣ Containerstart: FastAPI über Uvicorn
CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]