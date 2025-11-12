# Use official lightweight Python image
FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Install system deps for pillow/requests if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy app
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config: run on container start
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
