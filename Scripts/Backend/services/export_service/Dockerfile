FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gdal-bin libgdal-dev libkml-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV GDAL_DATA=/usr/share/gdal

COPY export_service_main.py .

CMD ["uvicorn", "export_service_main:app", "--host", "0.0.0.0", "--port", "8004"]
