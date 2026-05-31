# Dockerfile per Content Agent Pro su Railway
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install ffmpeg e dipendenze di sistema minime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copia i requisiti e installa le dipendenze Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia l'app
COPY . /app

# Esponi il porto standard e usa il cmd di default
EXPOSE 8000
CMD ["python", "content_agent_pro.py"]
