FROM python:3.10-slim

WORKDIR /app

# تثبيت المتطلبات النظامية
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# نسخ المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ التطبيق
COPY . .

# فتح المنفذ
EXPOSE 5000

# الأمر الافتراضي
CMD ["python", "app.py"]