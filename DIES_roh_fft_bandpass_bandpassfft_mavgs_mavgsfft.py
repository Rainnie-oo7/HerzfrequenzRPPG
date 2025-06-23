import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, butter, filtfilt, convolve
from collections import deque
import time
import os.path as osp

# === Polygon-Datenbank ===
saved_polygons = {
    "vid1.mp4": [[2, 267], [237, 287], [473, 279], [474, 587], [372, 574], [7, 620], [4, 299]],
    "vid2.mp4": [[35, 475], [121, 334], [126, 203], [152, 62], [179, 7], [580, 6], [551, 113], [518, 199], [517, 250], [491, 329], [413, 363], [321, 477], [60, 473]],
    "vid3.mp4": [[188, 477], [269, 320], [265, 186], [252, 98], [278, 9], [637, 4], [635, 242], [515, 343], [447, 452], [440, 474], [200, 478]],
    "vid4.mp4": [[177, 474], [280, 302], [278, 177], [303, 16], [310, 4], [637, 7], [635, 258], [526, 344], [479, 395], [448, 468], [186, 474]],
    "vid5.mp4": [[189, 434], [102, 247], [130, 43], [152, 4], [283, 6], [249, 76], [255, 102], [244, 184], [256, 201], [267, 191], [266, 114], [288, 53], [361, 53], [469, 113], [566, 227], [547, 280], [520, 378], [458, 461], [193, 436]],
    "vid6.mp4": [[117, 8], [217, 83], [403, 470], [634, 474], [636, 188], [492, 5], [121, 6]],
    "vid7.mp4": [[303, 477], [322, 142], [318, 79], [318, 3], [610, 5], [634, 205], [637, 477], [313, 477]]
}

# === Video einlesen ===
BASE_DIRECTORY = osp.normpath(osp.join(osp.dirname(__file__), "..", "..", "Pictures"))
VIDEO_FILENAME = "vid5.mp4"
video_path = osp.join(BASE_DIRECTORY, VIDEO_FILENAME)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Video konnte nicht geladen werden.")
    cap.release()
    exit()

# === Methode wählen ===
method = input("Welche Methode verwenden? (roh/fft/bandpass/bandpass+fft/mavg/mavg+fft): ").strip().lower()
if method not in ["roh", "fft", "bandpass", "bandpass+fft", "mavg", "mavg+fft"]:
    print("Ungültige Methode.")
    cap.release()
    exit()

# === Polygonwahl ===
polygon_points = []
antwort = input(f"Neues Polygon für '{VIDEO_FILENAME}' zeichnen? (j/n): ").strip().lower()

if antwort == "j":
    def draw_polygon(event, x, y, flags, param):
        global polygon_points, frame_copy
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
            frame_copy = frame.copy()
            for i in range(len(polygon_points)):
                cv2.circle(frame_copy, polygon_points[i], 3, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(frame_copy, polygon_points[i - 1], polygon_points[i], (0, 255, 0), 2)

    frame_copy = frame.copy()
    cv2.namedWindow("ROI wählen")
    cv2.setMouseCallback("ROI wählen", draw_polygon)

    while True:
        cv2.imshow("ROI wählen", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("y") and len(polygon_points) >= 3:
            break
        elif key == 27:
            print("Abbruch.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()
else:
    if VIDEO_FILENAME in saved_polygons:
        polygon_points = saved_polygons[VIDEO_FILENAME]
    else:
        print("Kein gespeichertes Polygon vorhanden.")
        cap.release()
        exit()

polygon = np.array(polygon_points, dtype=np.int32)
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [polygon], 255)

# === Funktionen ===
def bandpass_filter(signal, fps, low=0.7, high=3.0):
    nyq = 0.5 * fps
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def apply_hamming_filter(signal, window_size=21):
    if len(signal) < window_size:
        return signal
    hamming_window = np.hamming(window_size)
    hamming_window /= np.sum(hamming_window)
    filtered = convolve(signal, hamming_window, mode='valid')
    pad = np.full(len(signal) - len(filtered), filtered[0])
    return np.concatenate((pad, filtered))

def moving_average(signal, n):
    return np.convolve(signal, np.ones(n) / n, mode='valid')

# === Vorbereitung ===
sampling_rate = cap.get(cv2.CAP_PROP_FPS)
buffer_size = int(10 * sampling_rate)
intensity_buffer = deque(maxlen=buffer_size)
time_buffer = deque(maxlen=buffer_size)
bpm_over_time = []

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
line1, = ax1.plot([], [], label="Grün")
line2, = ax2.plot([], [], label=method.upper())

ax1.set_title("Grün-Intensität")
ax2.set_title("Herzfrequenz")
ax2.set_xlim(60, 165)
ax2.set_ylim(0, 1)

start_time = time.time()
last_10s_time = start_time
last_1s_time = start_time

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    roi = cv2.bitwise_and(frame, frame, mask=mask)
    green_mean = np.mean(roi[:, :, 1][mask == 255])
    now = time.time()
    current_time = now - start_time

    intensity_buffer.append(green_mean)
    time_buffer.append(current_time)

    times = np.array(time_buffer)
    signal = np.array(intensity_buffer)

    bpm_estimate = None
    if len(signal) >= sampling_rate * 5:
        if method == "roh":
            minima, _ = find_peaks(-signal, distance=sampling_rate * 0.5)
            duration = len(signal) / sampling_rate
            bpm_estimate = len(minima) / duration * 60
            ax2.clear()
            ax2.plot(times[-len(signal):], signal, color='green')
            if len(minima):
                ax2.plot(times[-len(signal):][minima], signal[minima], "ro")
            ax2.set_xlim(times.min(), times.max())
            ax2.set_ylim(signal.min() * 0.95, signal.max() * 1.05)
            ax2.set_title(f"rohe Intensität– scipy.signal.find_peaks BPM: {bpm_estimate:.1f}")
        elif method == "fft":
            N = len(signal)
            T = 1 / sampling_rate
            yf = fft(signal - np.mean(signal))
            xf = fftfreq(N, T)[:N // 2]
            bpm = xf * 60
            amplitude = 2.0 / N * np.abs(yf[0:N // 2])
            valid = (bpm >= 60) & (bpm <= 165)
            if np.any(valid):
                bpm_estimate = bpm[valid][np.argmax(amplitude[valid])]
                line2.set_data(bpm[valid], amplitude[valid])
                ax2.set_ylim(0, np.max(amplitude[valid]) * 1.1)
                ax2.set_title(f"FFT – Peak: {bpm_estimate:.1f} bpm")
        elif method == "bandpass":
            filtered = bandpass_filter(signal, sampling_rate)
            minima, _ = find_peaks(-filtered, distance=sampling_rate * 0.5)
            duration = len(filtered) / sampling_rate
            bpm_estimate = len(minima) / duration * 60
            ax2.clear()
            ax2.plot(times[-len(filtered):], filtered, color='green')
            if len(minima):
                ax2.plot(times[-len(filtered):][minima], filtered[minima], "ro")
            ax2.set_xlim(times.min(), times.max())
            ax2.set_ylim(filtered.min() * 0.95, filtered.max() * 1.05)
            ax2.set_title(f"Bandpass – BPM: {bpm_estimate:.1f}")
        elif method == "bandpass+fft":
            filtered = bandpass_filter(signal, sampling_rate)
            N = len(filtered)
            T = 1 / sampling_rate
            yf = fft(filtered - np.mean(filtered))
            xf = fftfreq(N, T)[:N // 2]
            bpm = xf * 60
            amplitude = 2.0 / N * np.abs(yf[0:N // 2])
            valid = (bpm >= 72) & (bpm <= 165)
            if np.any(valid):
                bpm_estimate = bpm[valid][np.argmax(amplitude[valid])]
                line2.set_data(bpm[valid], amplitude[valid])
                ax2.set_ylim(0, np.max(amplitude[valid]) * 1.1)
                ax2.set_title(f"Bandpass+FFT – Peak: {bpm_estimate:.1f} bpm")
        elif method == "mavg":
            n = 3
            filtered = moving_average(signal, n)
            minima, _ = find_peaks(-filtered, distance=sampling_rate * 0.5)
            duration = len(filtered) / sampling_rate
            bpm_estimate = len(minima) / duration * 60
            ax2.clear()
            ax2.plot(times[-len(filtered):], filtered, label="MA-2", color="blue")
            ax2.plot(times[-len(filtered):][minima], filtered[minima], "ro")
            ax2.set_xlim(times.min(), times.max())
            ax2.set_ylim(filtered.min() * 0.95, filtered.max() * 1.05)
            ax2.set_title(f"Moving Average neighbor frames = {n} rohe Intensität– BPM: {bpm_estimate:.1f}")
        elif method in ["mavg+fft"]:
            n = 3
            filtered = moving_average(signal, n)
            N = len(filtered)
            T = 1 / sampling_rate
            yf = fft(filtered - np.mean(filtered))
            xf = fftfreq(N, T)[:N // 2]
            bpm = xf * 60
            amplitude = 2.0 / N * np.abs(yf[0:N // 2])
            valid = (bpm >= 72) & (bpm <= 165)
            if np.any(valid):
                bpm_estimate = bpm[valid][np.argmax(amplitude[valid])]
                line2.set_data(bpm[valid], amplitude[valid])
                ax2.set_ylim(0, np.max(amplitude[valid]) * 1.1)
                ax2.set_title(f"Moving Average neighbor frames = {n}+FFT – Peak: {bpm_estimate:.1f} bpm")
    line1.set_data(times, signal)
    ax1.set_xlim(times.min(), times.max())
    ax1.set_ylim(signal.min() * 0.95, signal.max() * 1.05)

    if bpm_estimate and 72 <= bpm_estimate <= 165:
        if now - last_10s_time >= 10:
            print(f"[{int(current_time)}s] BPM (10s): {bpm_estimate:.1f}")
            last_10s_time = now
        if now - last_1s_time >= 1:
            bpm_over_time.append((current_time, bpm_estimate))
            last_1s_time = now

    plt.pause(0.001)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
plt.ioff()
plt.show()

# === Ausgabe ===
print("\n=== BPM-Werte alle 1 Sekunde ===")
for t, bpm in bpm_over_time:
    print(f"t = {t:.1f} s -> BPM = {bpm:.1f}")

if bpm_over_time:
    mean_bpm = np.mean([b for _, b in bpm_over_time])
    print(f"\n=== Gesamt-BPM: {mean_bpm:.1f} ===")
