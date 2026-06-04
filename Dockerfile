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
WORKDIR /app

# Install system dependencies (needed by DeepFace and standard Python libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the built React app from Stage 1
COPY --from=build /app/dist ./dist

# Copy the backend source code
COPY backend/ ./backend/

# Hugging Face Spaces expose port 7860 by default
EXPOSE 7860

# Start the unified server
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
