FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gee_service_main.py .

CMD ["uvicorn", "gee_service_main:app", "--host", "0.0.0.0", "--port", "8003"]
