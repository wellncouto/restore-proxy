FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# fonts pra marca d'água via Pillow + potrace/rsvg-convert pra vetorizar line art
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    potrace \
    librsvg2-bin \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py colorir.py ./

# storage persistente pros álbuns (montar volume aqui no Easypanel)
RUN mkdir -p /data/colorir
VOLUME ["/data/colorir"]

EXPOSE 8000

CMD ["python", "main.py"]
