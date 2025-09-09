# Use a slim Python image
FROM python:3.10-slim

# Prevents Python from writing .pyc files & enables unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set workdir
WORKDIR /app

# System deps (add as needed; keeping minimal for now)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . /app

# Expose port
ENV PORT=8080
EXPOSE 8080

# If your Flask app is in app.py and exposes `app`, this works:
# If your entry is different, e.g., chat_app.py with `app`, change to: gunicorn -b 0.0.0.0:$PORT chat_app:app
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

