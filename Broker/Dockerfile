FROM python:3.11-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variable for Kafka connection
ENV KAFKA_BROKER=kafka:29092

# Run the application
CMD ["python", "app.py"]

VOLUME [ "./models" ]