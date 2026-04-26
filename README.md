# SignSynth — Real-Time ASL Sign Language Detector

> Bridging the communication gap for the deaf and mute community using Computer Vision and Machine Learning.

---

## 🧠 What It Does

SignSynth detects American Sign Language (ASL) hand gestures in real time using a webcam or phone camera, translates them into text, and speaks them aloud. No internet required for detection. Built for the deaf and mute community as an accessible, low-cost communication tool.

---

## 👥 Team

**SignSynth** — ByteBattle 2026, Dr. DY Patil Technical Campus

| Name | Role |
|---|---|
| Vaibhav Shresth | ML Pipeline, MediaPipe Integration, App Development |
| Satyabrata Adak | Dataset Processing, Model Training, UI Design |

---

## 🚀 Features

- Real-time ASL alphabet detection (A–Z) via webcam
- 97.87% model accuracy on test set
- Text-to-speech output — speaks each letter and full words
- Blinking cursor word builder with history
- Live confidence bar and hold progress indicator
- Hand skeleton overlay with bounding box
- FPS control — cycle between 15, 30, 60 FPS
- Fullscreen dark UI with Rajdhani font
- Offline — runs entirely on local machine after setup
- Phone camera support via DroidCam (USB or WiFi)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Hand Detection | MediaPipe Hand Landmarker |
| ML Model | Random Forest (scikit-learn) |
| Camera Feed | OpenCV |
| UI Rendering | Pillow (PIL) |
| Text-to-Speech | gTTS + pygame / Windows SAPI |
| Language | Python 3.11 |

---

## 📁 Project Structure

```
signsynth/
  app.py                  ← main application (run this)
  extract_landmarks.py    ← extracts hand landmarks from dataset
  train_model.py          ← trains Random Forest classifier
  model.pkl               ← trained model (97.87% accuracy)
  hand_landmarker.task    ← MediaPipe hand detection model
  FONT.ttf                ← Rajdhani font
  requirements.txt        ← dependencies
```

---

## ⚙️ How It Works

```
Webcam Frame
     ↓
MediaPipe Hand Landmarker
     ↓
21 Hand Landmarks (x, y, z) → 63 features
     ↓
Random Forest Classifier
     ↓
Predicted Letter + Confidence Score
     ↓
Hold for N frames → Append to word
     ↓
Text-to-Speech Output
```

---

## 📊 Model Details

- **Dataset:** ASL Alphabet Dataset (Kaggle) — 87,000 images, 29 classes
- **Features:** 63 landmark coordinates per frame (21 points × x, y, z)
- **Model:** Random Forest, 100 estimators
- **Train/Test Split:** 80/20
- **Accuracy:** 97.87%
- **Classes:** A–Z, del, space, nothing

---

## 🔧 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/ShresthVaibhav/signsynth.git
cd signsynth
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
python app.py
```

---

## 📦 Requirements

```
mediapipe==0.10.33
opencv-python
scikit-learn
numpy
pandas
gtts
pygame
pillow
```

---

## 🎮 Controls

| Key | Action |
|---|---|
| SPACE | Speak current word aloud |
| C | Clear word and history |
| F | Cycle FPS (15 / 30 / 60) |
| Q | Quit |

---

## 📸 How to Use

1. Run `app.py`
2. Hold your hand in front of the camera
3. Sign an ASL letter and hold it steady
4. The letter appears on screen and is spoken aloud
5. Keep signing to build a word
6. Press SPACE to speak the full word
7. Press C to clear and start again

For best results:
- Use good lighting
- Keep hand fully visible in frame
- Plain background works best
- Hold each sign steady for ~0.5 seconds

---

## 🔭 Future Scope

- ISL (Indian Sign Language) support — dataset collection underway
- Two-hand sign support for word-level phrases
- Dynamic gesture recognition using LSTM for full words
- Mobile app using MediaPipe Android SDK + TFLite model
- Regional language TTS output (Hindi, Marathi)

---

## 🏆 Hackathon

Built for **ByteBattle 2026** at Dr. DY Patil Technical Campus, Varale-Talegaon, Pune.

Track: **AI & Machine Learning**

---

## 📄 License

MIT License — free to use, modify, and distribute.
