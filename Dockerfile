FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential ffmpeg poppler-utils && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
EXPOSE 7860
CMD ["python", "app.py"]
