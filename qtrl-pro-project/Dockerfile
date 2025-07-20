# Use a slim Python 3.11.9 base image for efficiency
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for qiskit, scipy, and alpaca-py
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and verify it exists
COPY requirements.txt .
RUN if [ ! -f requirements.txt ]; then echo "Error: requirements.txt not found" && exit 1; fi

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY qtrl_pro_trading.py .
COPY alpaca_data_service.py .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check for production
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run Streamlit app (default)
CMD ["streamlit", "run", "qtrl_pro_trading.py", "--server.port=8501", "--server.address=0.0.0.0"]