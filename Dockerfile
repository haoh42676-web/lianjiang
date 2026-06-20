FROM mcr.microsoft.com/playwright:v1.61.0-jammy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY package.json package-lock.json ./
RUN npm ci --omit=optional

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV LJ_NODE_PATH=/usr/bin/node

CMD ["python3", "ai_api_server.py"]
