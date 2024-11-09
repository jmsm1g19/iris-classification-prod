# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY docker_requirements.txt .
RUN pip install --no-cache-dir -r docker_requirements.txt && \
    pip install --no-cache-dir streamlit nest-asyncio && \
    python -m pip install --upgrade pip

# Copy the rest of the application
COPY . .

# Make start.sh executable
RUN chmod +x start.sh

# Expose ports for both FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Run the start.sh script
CMD ["./start.sh"]
