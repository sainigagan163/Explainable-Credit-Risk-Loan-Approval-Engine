FROM python:3.11-slim

WORKDIR /app

# Install supervisord
RUN apt-get update && apt-get install -y --no-install-recommends supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY credit_risk_engine/ credit_risk_engine/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000 8501

CMD ["./start.sh"]
