FROM python:3.11-slim

LABEL maintainer="ESG Alpha-Optimizer | Agentic Hedge Fund System"
LABEL description="Bank-grade local AI agent for ESG Alpha generation"

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download FinBERT model at build time (air-gapped after this)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; import os; model_path = '/workspace/models/finbert'; os.makedirs(model_path, exist_ok=True); print('Downloading FinBERT...'); AutoTokenizer.from_pretrained('ProsusAI/finbert', cache_dir=model_path); AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', cache_dir=model_path); print('FinBERT ready.')"

# Create audit log directory
RUN mkdir -p /workspace/audit_logs

# Copy application code
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]