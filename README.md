# VisionCount AI 👁‍🗨

VisionCount AI is a real-time, browser-based human detection and facial recognition counting system. It uses computer vision to detect humans in a room and utilizes deep learning facial recognition to ensure each person is uniquely identified and counted exactly once, even if they leave and return.

## Features

- **Real-time Human Detection:** Uses `tensorflow-models/coco-ssd` with a `mobilenet_v2` backbone for fast and accurate on-device body detection in the browser.
- **Facial Recognition Engine:** A FastAPI backend powered by `DeepFace` and `Facenet512`. Extracts 512-dimensional biometric embeddings to accurately distinguish between unique individuals.
- **Ghost Memory Tracking:** Remembers the last known position of people who temporarily step out of frame, reducing backend API calls and maintaining stability.
- **Strict De-duplication:** Built-in safeguards (NMS and IoU/IoM filtering) prevent nested or duplicate bounding boxes for the same person.
- **Premium UI:** A modern, glassmorphism-styled dark theme UI built with React, featuring glowing badges, live statistics, and counting animations.

## Tech Stack

### Frontend
- **Framework:** React + Vite
- **Detection:** TensorFlow.js (`@tensorflow/tfjs`) + COCO-SSD
- **Styling:** Vanilla CSS with modern CSS variables, glassmorphism, and responsive design

### Backend
- **Framework:** FastAPI (Python)
- **Facial Recognition:** DeepFace (`Facenet512` model) + OpenCV
- **Image Processing:** NumPy

## Prerequisites

- Node.js (v18+ recommended)
- Python (3.9+ recommended)

## Installation & Setup

### 1. Setup the Backend (Python)
Navigate to the `backend` directory and set up a virtual environment:

```bash
cd backend
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pydantic opencv-python numpy deepface tf-keras
```

Run the backend server:
```bash
python app.py
```
*(The backend runs on `http://localhost:8000`. On the first run, DeepFace will automatically download the required Facenet512 weights).*

### 2. Setup the Frontend (React)
Open a new terminal in the root directory of the project:

```bash
# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

The frontend will run on `http://localhost:5173/`. 
*(Note: Ensure you allow camera permissions when the browser prompts).*

## How it Works

1. **Detection:** The frontend webcam stream is analyzed by COCO-SSD. If a `person` is detected with > 50% confidence, a tracking box is drawn.
2. **Identification:** When a person is stable in the frame for a few seconds, their crop is sent to the FastAPI backend.
3. **Embedding Match:** The backend extracts a `Facenet512` embedding and compares it against known faces using cosine distance. 
    - If it's a match (distance < 0.20), the frontend logs them as `ALREADY DETECTED`.
    - If it's a new face, they are assigned a permanent ID, and the **Total Unique People** count increments by 1.

## License
MIT
