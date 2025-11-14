# Use Ubuntu base (Ollama needs Ubuntu/Debian)
FROM ubuntu:22.04

# Install essentials
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg curl git && \
    apt-get clean

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy project code
COPY . /app
WORKDIR /app

# Expose Flask port
EXPOSE 5000

# Start Ollama and Flask
CMD ollama serve & sleep 5 && python3 app.py
