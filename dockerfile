FROM debian:bullseye-slim

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip && apt-get clean

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]