
# 🚑 MedAI‑Vision

**MedAI‑Vision** is your all-in-one AI-powered medical imaging platform for detecting and analyzing health conditions from images and videos. Developed as a Final-Year Capstone project, it aims to streamline early diagnosis and patient monitoring using state-of-the-art computer vision and NLP techniques.

---

## 🧠 Features

- **Facial expression analysis** from video streams to detect signs of stress, fatigue, or emotional distress  
- **Symptom speech-to-text** conversion, followed by text-mining to flag potential depression symptoms  
- **Real-time video streaming** through a web app built with Flask  
- **User registration/authentication**, data logging with SQLite  
- **Results dashboard** to view historical emotion and text analysis outcomes  

---

## 🔧 Tech Stack

| Layer            | Tech |
|------------------|------|
| Web Framework     | Flask, Jinja2 |
| Frontend         | HTML, CSS, JavaScript |
| Video Processing | OpenCV, MoviePy |
| Audio Processing | SpeechRecognition |
| ML / NLP          | NLTK Naive Bayes |
| Database          | SQLite |
| Hosting           | Vercel (frontend), Flask backend on your own server |

---

## 📂 Repo Structure

```text
├── backend/        # Flask API & ML logic
├── frontend/       # Web UI templates & assets
├── models/         # Pre‑trained models (if used)
├── results/        # Sample output screenshots
├── uploads/        # Uploaded videos
├── README.md       # This file

🚀 Getting Started
Prerequisites
- **Python 3.x**
- **Node.js + npm (if setting up frontend separately)**
- **FFmpeg (optional, for better video/audio handling)**





