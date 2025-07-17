FROM python:3.11.8-slim-bullseye

# Sistem paketlerini güncelle ve gereksiz paketleri kaldır
RUN apt-get update && \
	apt-get upgrade -y && \
	apt-get install -y --no-install-recommends gcc && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# çalışma dizini belirle
WORKDIR /app

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# port aç (FastAPI default 8000)
EXPOSE 8000

# FastAPI uygulamasını başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
