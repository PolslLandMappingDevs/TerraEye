# --- STAGE 1: Build Frontend (Vite/React) ---
FROM node:18 AS build-step
WORKDIR /app/frontend

# Kopiujemy pliki konfiguracyjne frontendu
COPY io-app-front/package*.json ./
RUN npm install

# Kopiujemy resztę plików frontendu i budujemy (bez tsc, samo vite build)
COPY io-app-front/ ./
RUN npx vite build

# --- STAGE 2: Backend Server (Python + Flask) ---
# Używamy 3.12, żeby spełnić wymagania rasterio 1.5.0 i contourpy
FROM python:3.12-slim
WORKDIR /app

# Instalacja zależności systemowych dla GDAL i kompilacji bibliotek C++
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    python3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Zmienne środowiskowe dla kompilatora (kluczowe dla bibliotek GIS)
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Kopiujemy wymagania i instalujemy paczki
COPY io-app-backend/requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Przygotowanie struktury folderów dla backendu
WORKDIR /app/io-app-backend

# Kopiujemy zbudowany frontend (dist) do folderu static backendu
COPY --from=build-step /app/frontend/dist /app/io-app-backend/static

# Kopiujemy resztę kodu backendu
COPY io-app-backend/ .

# Konfiguracja Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Uruchomienie aplikacji
CMD ["python", "app.py"]
