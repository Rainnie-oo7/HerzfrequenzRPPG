import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import os.path as osp

def extract_ppg_signal(video_path, roi):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print("Fehler Video konnte nicht geöffnet werden.")
        return

    signal = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        green = roi_frame[:, :, 1]
        mean_val = green.mean()
        signal.append(mean_val)

    cap.release()
    return np.array(signal), fps

def bandpass_filter(signal, fps, low=0.7, high=3.0):  # 42–180 bpm
    nyq = 0.5 * fps #Nyquist minimale Länge
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def analyze_peaks(signal, fps):
    peaks, _ = find_peaks(signal, distance=fps*0.5)  # min 0.5 Sek Abstand
    duration = len(signal) / fps
    bpm = len(peaks) / duration * 60
    return bpm, peaks

# Videopfad:
BASE_DIRECTORY = osp.normpath(osp.join(osp.dirname(__file__), "..", "Pictures"))
VIDEO_FILENAME = "vid5.mp4"
video_path = osp.join(BASE_DIRECTORY, VIDEO_FILENAME)
roi = (250, 50, 340, 390)

signal_raw, fps = extract_ppg_signal(video_path, roi)
signal_filtered = bandpass_filter(signal_raw, fps)
bpm, peaks = analyze_peaks(signal_filtered, fps)

print(f"Geschätzte Herzfrequenz: {bpm:.1f} bpm")

# Plotting
plt.plot(signal_filtered, label="PPG (gefiltert)")
plt.plot(peaks, signal_filtered[peaks], "ro", label="Peaks")
plt.title("Photoplethysmographie-Signal")
plt.xlabel("Frame")
plt.ylabel("Intensität (Grün)")
plt.legend()
plt.show()

# vid1
# Geschätzte Herzfrequenz: 78.0 bpm
# vid2
# Geschätzte Herzfrequenz: 75.7 bpm
# vid3
# Geschätzte Herzfrequenz: 77.3 bpm