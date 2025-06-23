import cv2
import os.path as osp
# globale Variablen
drawing = False
ix, iy = -1, -1
rect = None  # (x1, y1, x2, y2)
frame_copy = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        rect = (x1, y1, x2, y2)
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Video laden und ersten Frame einfrieren ===
BASE_DIRECTORY = osp.normpath(osp.join(osp.dirname(__file__), "..", "Pictures"))
VIDEO_FILENAME = "vid6.mp4"
video_path = osp.join(BASE_DIRECTORY, VIDEO_FILENAME)
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Konnte Video nicht laden.")
    cap.release()
    exit()

frame_copy = frame.copy()
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', draw_rectangle)

# Interaktive Anzeige
while True:
    cv2.imshow('Frame', frame_copy)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('y') or key == ord('Y'):
        if rect:
            x1, y1, x2, y2 = rect
            print(f"{y1}:{y2}, {x1}:{x2}")
        else:
            print("Noch kein Rechteck gezeichnet.")

    elif key == 27:  # ESC zum Beenden
        break

cv2.destroyAllWindows()
cap.release()

"""
284:579, 2:478  vid1 linker Unterarm Oberseite  - Boot - Hochkant
161:565, 238:479 vid2 linke Hand Oberseite       - Hochhaus - Querkant GEDREHT25
138:386, 322:632 vid3 linke Hand Oberseite       - Boot - Hochkant GEDREHT30
3:278, 300:637 vid4 linke Hand Oberseite       - Boot - Querkant GEDREHT28
113:421, 172:497 vid5 Linke Hand Innenseite      - Boot - Hochkant
vid6 rechter Unterarm Oberseite - Hochhaus - Querkant GEDREHT -30
vid7 Zaid's linker Unterarm Oberseite - Hochhaus - Querkant
"""