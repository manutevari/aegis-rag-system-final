FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl libpq-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/policies
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser
EXPOSE 8000 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
