# Dockerfile

# 1. Choose a base Python image
FROM python:3.11-slim AS base

# 2. Set environment variables
# Prevents Python from writing pyc files to disc (equivalent to python -B)
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr (important for logging)
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install dependencies
# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .
# Install system dependencies that might be needed by some Python packages (optional, add if needed)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the application code into the container
# Copy the rest of the application code (including your Streamlit script, assuming it's named app.py)
COPY . .

# 6. Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# 7. Define the command to run the application
# Run Streamlit.
# --server.address=0.0.0.0 allows it to be accessible from outside the container.
# --server.enableCORS=false and --server.enableXsrfProtection=false might be needed depending on your setup/ingress,
# but start without them unless you face issues.
CMD ["streamlit", "run", "chatbot_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
