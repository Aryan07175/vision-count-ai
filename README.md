---
title: Vision Count AI
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 👁️ VisionCount AI

Hey there! Welcome to **VisionCount AI**. 

I built this project to solve a really annoying problem with most people-counting cameras: **double counting**. If someone walks into a room, leaves for a minute, and comes back, most systems count them twice. 

VisionCount fixes this by giving everyone a unique "Face-ID" using artificial intelligence. It runs right in your web browser!

🚀 **[Try the Live Demo Here!](https://aaru07160-vision-count-ai.hf.space/)** 🚀

*(Note: When you open the live demo, make sure to click "Allow" when your browser asks for camera access!)*

---

## ✨ What makes it cool?

- **Real-Time Tracking:** It uses your webcam to spot people instantly as they walk by.
- **Smart Memory:** It remembers your face! If you duck out of the camera's view and pop back in, it knows it's still you. No double counting.
- **Privacy First:** All the facial recognition happens locally or on a secure private server. 
- **Sleek Interface:** I spent a lot of time making the dashboard look clean, dark, and futuristic. 

---

## 🛠️ How it works (The simple version)

The app is split into two parts working together:

1. **The Web Browser (Frontend):** 
   Built with **React**. This handles the webcam and draws those cool tracking boxes around people. It uses a lightweight AI (TensorFlow.js) to say, "Hey, I see a human shape!"

2. **The Brains (Backend):**
   Built with **Python & FastAPI**. When the browser spots a human, it takes a quick snapshot and sends it here. The backend uses a heavier AI (DeepFace) to look at the face and say, "Oh, that's Person #3, we've seen them before."

## 💻 Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Frontend UI** | React 19, Vite, Capacitor |
| **In-Browser AI (Detection)** | TensorFlow.js, COCO-SSD |
| **Backend API** | Python, FastAPI, Uvicorn |
| **Server-side AI (Recognition)** | DeepFace, OpenCV, NumPy |

---

## 🚀 Want to run it yourself?

If you want to download the code and play with it on your own computer, here is how you do it!

### 1. Download the code
```bash
git clone https://github.com/Aryan07175/vision-count-ai.git
cd vision-count-ai
```

### 2. Start the AI Backend (Python)
You'll need Python installed on your computer.
```bash
# Go to the backend folder
cd backend

# Create a virtual environment so we don't mess up your computer's Python
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate
# (Or on Mac/Linux: source venv/bin/activate)

# Install the AI libraries
pip install -r requirements.txt

# Start the server!
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
*(Pro tip: The very first time you run this, it will take a minute or two to download the AI models from the internet. Just let it do its thing!)*

### 3. Start the Website (React)
Open a **new** terminal window and go back to the main project folder.
```bash
# Install the web packages
npm install

# Start the website
npm run dev
```

That's it! Now just open `http://localhost:5173` in your browser, step in front of your webcam, and watch it count you!

---

## 🔧 Troubleshooting

- **The camera is just a black box?** Make sure you clicked "Allow" for camera permissions in your browser. Also, close any other apps (like Zoom) that might be using your webcam.
- **The counter isn't going up?** Make sure your Python backend is running without errors. If the backend is turned off, the website will switch to "Local Mode" and might struggle to remember faces accurately.

## 📄 License
This project is completely free and open-source under the MIT License. Feel free to use it, change it, and make it your own!
