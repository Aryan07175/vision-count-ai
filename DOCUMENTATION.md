# VisionCount AI: Technical Documentation

## 1. Executive Summary
VisionCount AI is an advanced, real-time computer vision system designed to accurately detect, track, and count human individuals in a live camera feed. By combining edge-based object detection with server-side facial recognition, the system ensures high accuracy and eliminates common tracking errors such as double-counting.

## 2. Problem Statement
Traditional people-counting systems rely heavily on simple motion detection, line-crossing algorithms, or basic bounding-box tracking. These legacy approaches suffer from critical flaws:
* **Double-Counting on Re-entry:** If an individual leaves the camera's field of view and returns, they are erroneously counted as a new person.
* **Occlusion Failures:** When individuals cross paths or temporarily block each other, basic tracking algorithms lose the subject's ID, resulting in miscounts.
* **Lack of Biometric Persistence:** Without facial recognition, the system has no long-term memory of unique individuals over an extended period.

## 3. Solution Architecture
To solve the aforementioned problems, VisionCount AI employs a **Two-Layered Hybrid Architecture**:

### 3.1. Edge-Based Tracking (Frontend)
The web browser acts as the edge device, running a lightweight machine learning model (`COCO-SSD` via TensorFlow.js). 
* **High-Frequency Detection:** The browser scans the live webcam feed every 150 milliseconds to detect human bodies.
* **Ghost Memory:** The frontend maintains a spatial memory registry. If a tracked bounding box disappears, it is retained as a "ghost" for up to 3 seconds, gracefully handling temporary occlusions without triggering a new count.

### 3.2. Biometric Verification (Backend)
When the frontend establishes a stable, unrecognized human track, it captures a cropped image of the subject's face and transmits it to the Python backend.
* **Facial Embedding:** The backend utilizes `DeepFace` (powered by Facenet512) to extract a 512-dimensional biometric embedding of the face.
* **Similarity Matching:** The new embedding is mathematically compared against an in-memory database of all previously seen individuals. If the facial similarity exceeds the confidence threshold, the backend returns the existing ID, preventing a double count. If it is a new face, a permanent unique ID is assigned.

## 4. Technology Stack
The project is built upon a modern, full-stack web architecture:

### Frontend
* **Core Framework:** React 18, Vite
* **Machine Learning:** TensorFlow.js, COCO-SSD (MobileNetV2)
* **Media Handling:** `react-webcam` API for secure hardware access
* **Styling:** Custom CSS featuring a dark-themed, glassmorphic design system

### Backend
* **Core Framework:** Python 3.10+, FastAPI, Uvicorn
* **Machine Learning:** DeepFace, TensorFlow (tf-keras), OpenCV Headless
* **Data Handling:** NumPy, Pydantic

### Infrastructure & Deployment
* **Containerization:** Multi-stage Docker build integrating both the Node.js frontend and Python backend into a single unified image.
* **Hosting Environment:** Hugging Face Spaces (Docker SDK), operating in a restricted-user (UID 1000) Linux environment.

## 5. Development & Deployment History
During the recent development sprints, the following critical improvements and architectural changes were implemented:

1. **System Stability & WebGL Optimization:** 
   * Resolved a critical issue where the browser's WebGL context would crash the camera feed under heavy load. The webcam feed resolution was optimized to 640x480, significantly reducing GPU memory consumption and preventing video freezes.
2. **Asynchronous API Resiliency:** 
   * Addressed frequent "Backend Offline" false positives. The HTTP timeout threshold for the DeepFace inference API was extended from 6 seconds to 15 seconds, accommodating slower CPU environments.
3. **Unified Deployment Architecture:** 
   * Designed a multi-stage `Dockerfile` that automatically compiles the React frontend and mounts the static asset bundle directly into the FastAPI server. This eliminated Cross-Origin Resource Sharing (CORS) complexities and allowed for single-server hosting.
4. **Linux Headless Compatibility:** 
   * Replaced standard OpenCV bindings with `opencv-python-headless` and explicitly injected necessary system libraries (`libgl1`, `libglib2.0-0`) into the Docker environment, ensuring robust execution in Debian-based cloud environments.
5. **Hugging Face Cloud Deployment:** 
   * Configured the necessary YAML metadata and system user permissions (UID 1000) to successfully deploy the unified application to Hugging Face Spaces, making the AI system globally accessible via a public URL.
