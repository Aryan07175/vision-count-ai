# Stage 1: Build the React frontend
FROM node:20-alpine AS build
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy all source files and build
COPY . .
RUN npm run build

# Stage 2: Setup the Python backend
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces require running as a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy backend requirements and install as user
COPY --chown=user backend/requirements.txt ./backend/
RUN pip install --user --no-cache-dir -r backend/requirements.txt

# Copy the built React app from Stage 1
COPY --chown=user --from=build /app/dist ./dist

# Copy the backend source code
COPY --chown=user backend/ ./backend/

# Hugging Face Spaces expose port 7860 by default
EXPOSE 7860

# Start the unified server
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
