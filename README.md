# SignSynth — Real-Time ASL Sign Language Detector

> Bridging the communication gap for the deaf and mute community using Computer Vision and Machine Learning.

![Architecture](SignSynth_Architecture.png)

---

## 👥 Team

| Name | Role |
|---|---|
| Vaibhav Shresth | ML Pipeline, MediaPipe Integration, App Development & UI |
| Satyabrata Adak | Dataset Processing, Model Training, Testing & Validation |

**ByteBattle 2026 · Track 1: AI & Machine Learning · Dr. DY Patil Technical Campus**

**GitHub:** https://github.com/ShresthVaibhav/SignSynth

---

## 📌 Project Abstract

SignSynth is a real-time desktop application that detects American Sign Language (ASL) hand gestures through a standard webcam, translates them into text, and speaks them aloud using text-to-speech — with no internet connection required after setup. It runs entirely on local hardware using a trained Machine Learning model, making it accessible even in low-connectivity environments.

---

## ❌ Problem Statement

Over **63 million people** in India live with significant hearing loss. Deaf and mute individuals face severe communication barriers every single day — at hospitals, government offices, banks, and classrooms — because hearing people do not understand sign language.

Current workarounds are slow and broken:

- **Pen and paper** — slow, requires literacy, no audio, breaks conversation flow
- **Phone typing** — both parties share a device, slow, no TTS, battery dependent
- **Human interpreter** — expensive, not always available, privacy concerns

No affordable, real-time, software-based solution exists for Indian users today.

---

## ✅ Our Solution

SignSynth detects hand signs in real time, converts them to text, and speaks them aloud. A deaf person signs in front of any webcam — the word appears on screen and is spoken through speakers. Hearing people understand immediately without learning any sign language.

### Why SignSynth is better

| Feature | Paper / Phone | SignSynth |
|---|---|---|
| Speed | Slow, one word at a time | Real-time, continuous |
| Audio output | None | Built-in TTS |
| Requires shared device | Yes (phone) | No |
| Works offline | Yes | Yes |
| Natural conversation flow | No | Yes |
| Scalable to more hardware | No | Yes — smart glasses, kiosks |

---

## 📊 Key Results

| Model Accuracy | Sign Classes | Landmark Features | Inference Time |
|---|---|---|---|
| **97.87%** | **29** | **63** | **<50ms** |

---

## 🔬 How It Works — Technical Pipeline

```
Webcam Frame
     ↓
MediaPipe Hand Landmarker (Google)
     ↓
21 Hand Landmarks (x, y, z) → 63-feature vector
     ↓
Random Forest Classifier (trained by us)
     ↓
Predicted Letter + Confidence Score
     ↓
Hold logic → Word builder → TTS Output
```

| Stage | Technology | Description |
|---|---|---|
| Camera Input | OpenCV | Captures frames at 15/30/60 FPS (selectable) |
| Hand Detection | MediaPipe Hand Landmarker | Extracts 21 landmark points per hand |
| Feature Vector | NumPy | 21 pts × 3 coords = 63 normalized features |
| Classification | scikit-learn Random Forest | 100 estimators, 97.87% test accuracy |
| TTS Output | gTTS + pygame / Windows SAPI | Speaks letters and words aloud |
| UI | OpenCV + Pillow | Fullscreen dark UI with custom font |

---

## 🛠️ List of Technologies & Libraries

| Library / Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | Primary language |
| MediaPipe | 0.10.33 | Hand landmark detection |
| OpenCV | Latest | Camera capture, frame processing |
| scikit-learn | Latest | Random Forest training and inference |
| NumPy | Latest | Feature vector construction |
| Pandas | Latest | Dataset loading and preprocessing |
| Pillow (PIL) | Latest | UI rendering with TTF font |
| gTTS | Latest | Online text-to-speech |
| pygame | Latest | Audio playback |
| Windows SAPI | Built-in | Offline TTS fallback |

---

## 🤖 AI Tools & Open Source Disclosure

*As required by ByteBattle 2026 Code of Conduct — full disclosure of AI tool usage.*

### Open Source Libraries Used
All open-source libraries are credited in the tech stack table above. MediaPipe Hand Landmarker is a pre-trained model by Google used as a feature extractor — its weights were not modified.

### AI Tools Used

**Claude (Anthropic)** was used during development for the following:

- **Dataset research** — identifying and evaluating suitable ASL datasets across Kaggle, HuggingFace, and Mendeley for image count, label quality, and compatibility with our pipeline
- **Architecture decisions** — researching CNN vs landmark-based approach; landmark approach chosen for lower compute, faster inference, and better real-world generalization
- **Debugging** — resolving MediaPipe v0.10 breaking API changes that affected older community solutions
- **UI design guidance** — structuring the PIL-based fullscreen rendering pipeline
- **Documentation** — assistance in writing README, abstract, and project documentation

### What we built ourselves
All core logic was written, understood, and modified by the team:
- Landmark extraction pipeline (`extract_landmarks.py`)
- Model training and evaluation (`train_model.py`)
- Hold/cooldown detection system for letter registration
- Full application UI with custom font rendering (`app.py`)
- TTS integration with online/offline fallback

No code was directly copied without understanding or modification.

---

## 🌍 Real-World Applications & Impact

**Assistive Communication** — deaf and mute individuals communicating with hearing people in hospitals, offices, banks, and schools without needing a human interpreter.

**Smart Glasses Integration** — SignSynth's lightweight pipeline can run on edge hardware. Integrated into smart glasses, it translates signs in the user's field of view and speaks through a bone conduction speaker — completely hands-free real-time communication.

**Classroom Accessibility** — a mounted camera detects a deaf student's signs and displays them on the teacher's screen in real time, eliminating the need for a dedicated interpreter in every classroom.

**Telemedicine** — during video consultations, a deaf patient signs to the camera and the system translates to text visible to the doctor — no third-party interpreter required.

**Public Kiosks** — government and banking self-service kiosks with sign language detection as a built-in accessibility feature, enabling deaf users to interact independently.

**Emergency Services** — deaf individuals contacting emergency services via video could have their signs translated to text for dispatchers in real time, significantly improving emergency response.

---

## 📈 Progress Report

| Milestone | Status |
|---|---|
| Problem research & dataset selection | ✅ Done |
| MediaPipe landmark extraction pipeline | ✅ Done |
| ASL dataset processing (87,000 images) | ✅ Done |
| Random Forest model training | ✅ Done — 97.87% accuracy |
| Live camera detection app | ✅ Done |
| Text-to-speech integration | ✅ Done |
| Custom dark UI with font rendering | ✅ Done |
| Phone camera support (DroidCam) | ✅ Done |
| GitHub repository | ✅ Done |
| Documentation & abstract | ✅ Done |
| ISL (Indian Sign Language) support | 🔄 In progress |
| Two-hand phrase detection | 🔄 Planned |
| Mobile app (TFLite) | 🔄 Planned |

---

## 📁 Project Structure

```
SignSynth/
  app.py                  ← main application (run this)
  extract_landmarks.py    ← extracts hand landmarks from dataset
  train_model.py          ← trains Random Forest classifier
  model.pkl               ← trained model (97.87% accuracy)
  hand_landmarker.task    ← MediaPipe hand detection model
  FONT.ttf                ← UI font (Rajdhani)
  requirements.txt        ← all dependencies
  README.md               ← this file
```

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/ShresthVaibhav/SignSynth.git
cd SignSynth
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

## 🎮 Controls

| Key | Action |
|---|---|
| SPACE | Speak current word aloud |
| C | Clear word and history |
| F | Cycle FPS (15 / 30 / 60) |
| Q | Quit |

---

## 📸 Tips for Best Results

- Good lighting — avoid dark or heavily backlit environments
- Keep hand fully visible in frame at all times
- Plain background gives better detection accuracy
- Hold each sign steady for ~0.5 seconds before it registers
- Use your dominant hand

---

## 🔭 Future Scope

- ISL (Indian Sign Language) support — dataset collection underway
- Two-hand sign detection for word-level phrases
- Dynamic gesture recognition using LSTM for full word signing
- Mobile app using MediaPipe Android SDK + TFLite
- Regional language TTS output (Hindi, Marathi)
- Smart glasses hardware integration

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

*Built for ByteBattle 2026 · Dr. DY Patil Technical Campus, Varale-Talegaon, Pune 410507*
*Contact: bytebattle.dyptc@parvadable.com*
