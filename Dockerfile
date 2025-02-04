FROM python:3.9-slim as backend

# Set working directory for the backend
WORKDIR /app

# Copy backend requirements
COPY app/requirements.txt ./

# Install backend dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY app/ ./ .env ./

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
