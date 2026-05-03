FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# fonts pra marca d'água via Pillow + potrace/rsvg-convert pra vetorizar line art
# + libs pro Chromium do Playwright (deps apt instaladas manualmente, sem sudo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    potrace \
    librsvg2-bin \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libasound2 libatspi2.0-0 libxkbcommon0 libwayland-client0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chromium pra renderizar capa via HTML (Playwright). Sem --with-deps (apt já tá feito acima)
RUN playwright install chromium

COPY main.py colorir.py ./
COPY fonts ./fonts
COPY templates ./templates

# storage persistente pros álbuns (montar volume aqui no Easypanel)
RUN mkdir -p /data/colorir
VOLUME ["/data/colorir"]

EXPOSE 8000

CMD ["python", "main.py"]
