# ==================== FOOTFALL COUNTER — VS CODE VERSION ====================
# ✅ Accepts any video file name
# ✅ YOLOv8 + DeepSORT tracking
# ✅ Live ENTRY/EXIT counters
# ✅ Final TOTAL COUNT frame added for 2 seconds
# ✅ Output always: output.mp4
# ============================================================================

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# ---------------- Ask user for video name ----------------
INPUT_VIDEO = input("Enter your input video filename (example: mall.mp4): ").strip()
OUTPUT_VIDEO = "output.mp4"

if not os.path.exists(INPUT_VIDEO):
    raise FileNotFoundError(f"Video file '{INPUT_VIDEO}' not found in this folder.")

print(f"\n✅ Input: {INPUT_VIDEO}")
print(f"✅ Output will be: {OUTPUT_VIDEO}\n")

# ---------------- Load YOLO ----------------
model = YOLO("yolov8n.pt")

# ---------------- Init DeepSORT ----------------
tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_iou_distance=0.7,
    embedder="mobilenet",
    half=True,
    bgr=True,
)

# ---------------- Video setup ----------------
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W, H))

# ---------------- Counting ----------------
entry_count = 0
exit_count  = 0
last_side = {}

line_y = int(0.55 * H)

def get_side(pt):
    return -1 if pt[1] < line_y else 1

print("Processing...")

last_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    last_frame = frame.copy()

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
        conf       = float(box.conf[0])
        detections.append(([x1,y1,x2-x1,y2-y1], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        x1,y1,x2,y2 = map(int, track.to_ltrb())
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        side = get_side((cx, cy))

        if tid not in last_side:
            last_side[tid] = side
        else:
            prev = last_side[tid]
            if prev != side:
                if prev == -1 and side == 1:
                    entry_count += 1
                elif prev == 1 and side == -1:
                    exit_count += 1
            last_side[tid] = side

        cv2.rectangle(frame, (x1,y1), (x2,y2), (50,180,255), 2)
        cv2.putText(frame, f"ID {tid}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.circle(frame, (cx,cy), 4, (255,255,255), -1)

    cv2.line(frame, (0,line_y), (W,line_y), (0,255,255), 3)
    cv2.putText(frame, f"ENTRY: {entry_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"EXIT:  {exit_count}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

    writer.write(frame)

cap.release()

# ---------------- FINAL SUMMARY FRAME ----------------
total = entry_count + exit_count
summary = last_frame.copy()

cv2.rectangle(summary, (30,30), (580,260), (0,0,0), -1)
cv2.putText(summary, "FINAL FOOTFALL SUMMARY", (50,90),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
cv2.putText(summary, f"ENTRY COUNT : {entry_count}", (50,150),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
cv2.putText(summary, f"EXIT COUNT  : {exit_count}", (50,190),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,165,255), 2)
cv2.putText(summary, f"TOTAL COUNT : {total}", (50,230),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,0), 3)

for _ in range(int(fps * 2)):
    writer.write(summary)

writer.release()

print("\n✅ COMPLETED SUCCESSFULLY!")
print("✅ ENTRY :", entry_count)
print("✅ EXIT  :", exit_count)
print("✅ TOTAL :", total)
print("✅ Output saved as output.mp4")
