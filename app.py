import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import os
import time
import warnings
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')

# ================= FONT =================
FONT_PATH   = 'FONT.ttf'
font_title  = ImageFont.truetype(FONT_PATH, 45)
font_big    = ImageFont.truetype(FONT_PATH, 60)
font_med    = ImageFont.truetype(FONT_PATH, 30)
font_small  = ImageFont.truetype(FONT_PATH, 25)
font_tiny   = ImageFont.truetype(FONT_PATH, 20)

# ================= COLORS =================
C_ACCENT  = (100, 180, 255)
C_GREEN   = (80,  220, 120)
C_YELLOW  = (255, 210, 80)
C_WHITE   = (240, 240, 245)
C_GRAY    = (120, 120, 135)
C_RED     = (255, 80,  80)
C_PANEL   = (14,  14,  22)
C_BORDER  = (40,  60,  90)

def cc(conf):
    if conf >= 0.7:   return (80, 220, 120)
    elif conf >= 0.4: return (255, 200, 60)
    return (255, 100, 80)

def txt(draw, text, pos, font, color):
    draw.text(pos, text, font=font, fill=color)

def panel(draw, x, y, w, h):
    draw.rectangle([x, y, x+w, y+h], fill=C_PANEL, outline=C_BORDER, width=1)

def bar(draw, x, y, w, h, ratio, color):
    draw.rectangle([x, y, x+w, y+h], fill=(35, 35, 50))
    if ratio > 0:
        draw.rectangle([x, y, x+int(w*ratio), y+h], fill=color)
    draw.rectangle([x, y, x+w, y+h], outline=C_BORDER, width=1)

# ================= AUDIO =================
def speak(text):
    def run():
        try:
            from gtts import gTTS
            import pygame, tempfile
            pygame.mixer.init()
            tts = gTTS(text=text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                tts.save(f.name); fname = f.name
            time.sleep(0.1)
            pygame.mixer.music.load(fname)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): pygame.time.wait(100)
            pygame.mixer.music.unload(); time.sleep(0.1); os.unlink(fname)
        except Exception:
            os.system(f'PowerShell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"')
    threading.Thread(target=run, daemon=True).start()

# ================= MODEL =================
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ================= MEDIAPIPE =================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

FINGER_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ================= STATE =================
word        = ''
prev_letter = ''
hold_count  = 0
HOLD_THRESHOLD  = 12
cooldown        = 0
COOLDOWN_FRAMES = 20
prev_time       = 0
fps_options     = [15, 30, 60]
fps_idx         = 1
history         = []

# ================= WINDOW =================
cv2.namedWindow('SignSynth', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('SignSynth', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print('SignSynth ready | SPACE: Speak | C: Clear | F: FPS | Q: Quit')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1920, 1080))
    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]

    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    letter        = ''
    confidence    = 0.0
    hand_detected = False

    if result.hand_landmarks:
        hand_detected = True
        landmarks     = result.hand_landmarks[0]

        xs = [int(lm.x * W) for lm in landmarks]
        ys = [int(lm.y * H) for lm in landmarks]

        for conn in FINGER_CONN:
            pt1 = (int(landmarks[conn[0]].x * W), int(landmarks[conn[0]].y * H))
            pt2 = (int(landmarks[conn[1]].x * W), int(landmarks[conn[1]].y * H))
            cv2.line(frame, pt1, pt2, (60, 120, 200), 2)

        for lm in landmarks:
            cv2.circle(frame, (int(lm.x*W), int(lm.y*H)), 5, (100, 200, 255), -1)

        bx1 = max(0, min(xs)-20); by1 = max(0, min(ys)-20)
        bx2 = min(W, max(xs)+20); by2 = min(H, max(ys)+20)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (100, 180, 255), 2)

        row = []
        for lm in landmarks:
            row += [lm.x, lm.y, lm.z]

        letter = model.predict([row])[0]
        if hasattr(model, 'predict_proba'):
            confidence = float(np.max(model.predict_proba([row])))

        if letter == prev_letter:
            hold_count += 1
        else:
            hold_count  = 0
            prev_letter = letter

        if cooldown > 0:
            cooldown -= 1
        elif hold_count >= HOLD_THRESHOLD:
            if letter == 'space':
                word += ' '; speak('space')
            elif letter == 'del':
                word = word[:-1]; speak('delete')
            elif letter != 'nothing':
                word += letter; speak(letter)
                history.append(letter)
                if len(history) > 6: history.pop(0)
            hold_count = 0
            cooldown   = COOLDOWN_FRAMES

    # ================= PIL UI =================
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d   = ImageDraw.Draw(img)

    curr_time = time.time()
    fps_real  = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # ── TOP BAR ───────────────────────────────────────────
    d.rectangle([0, 0, W, 50], fill=(10, 10, 16))
    d.line([(0,44),(W,44)], fill=C_BORDER, width=1)
    txt(d, 'SignSynth', (16, 3), font_title, C_ACCENT)

    # hand status
    dot_c = C_GREEN if hand_detected else C_RED
    d.ellipse([W-310, 12, W-292, 30], fill=dot_c)
    txt(d, 'Hand Detected' if hand_detected else 'No Hand', (W-284, 8), font_small, dot_c)

    # fps — separated far right
    txt(d, f'{int(fps_real)} / {fps_options[fps_idx]} fps', (W-90, 8), font_tiny, C_GRAY)

    # ── TOP LEFT STATS ────────────────────────────────────
    sx, sy, sw, sh = 10, 54, 190, 160
    panel(d, sx, sy, sw, sh)
    txt(d, 'STATS', (sx+8, sy+5), font_tiny, C_GRAY)
    d.line([(sx+8, sy+27),(sx+sw-8, sy+29)], fill=C_BORDER, width=1)

    txt(d, 'Confidence', (sx+8, sy+26), font_tiny, C_GRAY)
    bar(d, sx+8, sy+50, sw-16, 7, confidence, cc(confidence))
    txt(d, f'{confidence*100:.0f}%', (sx+8, sy+61), font_tiny, cc(confidence))

    txt(d, 'Hold Progress', (sx+8, sy+78), font_tiny, C_GRAY)
    bar(d, sx+8, sy+102, sw-16, 7, min(hold_count/HOLD_THRESHOLD,1.0), C_ACCENT)
    txt(d, f'{hold_count} / {HOLD_THRESHOLD}', (sx+8, sy+113), font_tiny, C_WHITE)

    txt(d, f'Target  {fps_options[fps_idx]} fps  [F]', (sx+8, sy+132), font_tiny, C_ACCENT)

    # ── BOTTOM LEFT — DETECTED SIGN ───────────────────────
    dx, dy, dw, dh = 10, H-170, 170, 140
    panel(d, dx, dy, dw, dh)
    txt(d, 'Detected Sign', (dx+8, dy+6), font_tiny, C_GRAY)
    d.line([(dx+8, dy+20),(dx+dw-8, dy+20)], fill=C_BORDER, width=1)

    sign_str = letter.upper() if letter else '·'
    sign_col = C_GREEN if letter else C_GRAY
    txt(d, sign_str, (dx+14, dy+24), font_big, sign_col)

    txt(d, f'{confidence*100:.0f}% conf', (dx+8, dy+88), font_tiny, cc(confidence))
    txt(d, '  '.join(history) if history else '· · ·', (dx+8, dy+108), font_tiny, C_GRAY)

    # ── BOTTOM — OUTPUT BOX ───────────────────────────────
    ox, oy = 190, H - 100
    ow, oh = W - 200, 74
    panel(d, ox, oy, ow, oh)
    txt(d, 'Output', (ox+12, oy+5), font_tiny, C_GRAY)
    d.line([(ox+12, oy+20),(ox+ow-12, oy+20)], fill=C_BORDER, width=1)

    cursor       = '|' if int(time.time()*2) % 2 == 0 else ' '
    word_display = word[-60:] + cursor if len(word) > 60 else word + cursor
    txt(d, word_display, (ox+14, oy+28), font_med, C_YELLOW)

    # ── BOTTOM BAR ────────────────────────────────────────
    d.rectangle([0, H-22, W, H], fill=(10,10,16))
    d.line([(0, H-22),(W, H-22)], fill=C_BORDER, width=1)
    txt(d, 'SPACE  Speak Word          C  Clear          F  Cycle FPS          Q  Quit',
        (20, H-18), font_tiny, C_GRAY)

    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow('SignSynth', frame)

    key = cv2.waitKey(max(1, int(1000/fps_options[fps_idx]))) & 0xFF
    if key == ord('q'):   break
    elif key == ord(' '):
        if word.strip(): speak(word)
    elif key == ord('c'): word = ''; history = []
    elif key == ord('f'): fps_idx = (fps_idx+1) % len(fps_options)

cap.release()
cv2.destroyAllWindows()
