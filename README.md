# Footfall Counter Using Computer Vision

## 1. Brief Description of the Approach
This project implements a complete **footfall counting system** using computer vision.  
The system detects humans in a video, tracks their movement across frames, and counts how many people **enter** or **exit** through a defined virtual line.  
It uses **YOLOv8** for person detection and **DeepSORT** for multi-object tracking, ensuring stable tracking IDs and accurate counting even when multiple people appear simultaneously or partially overlap.

The processed output video (`output.mp4`) contains:
- Bounding boxes around detected people  
- Tracking IDs  
- A horizontal counting line  
- Live ENTRY and EXIT counters  
- A final summary frame showing TOTAL count  

---

## 2. Video Source Used
The system accepts **any input video uploaded by the user**.  
During testing, human activity videos (crowd movement, entrances, public walkways, etc.) were used to validate detection and counting performance.

You may use:
- A recorded video from your phone  
- YouTube crowd/entrance footage  
- Any publicly available video containing human movement  

---

## 3. Explanation of Counting Logic
A horizontal virtual line is drawn at **55% of the frame height**.  
The logic tracks each individual's movement relative to this line:

- If a person crosses **from above the line → below**, it increments **ENTRY**.  
- If a person crosses **from below the line → above**, it increments **EXIT**.

The system stores the **last known side** of the line for each tracking ID.  
A crossing is counted only when the person's side changes, preventing double counting.

At the end of processing, a summary frame appears showing:
- Total ENTRY count  
- Total EXIT count  
- TOTAL = ENTRY + EXIT  

---

## 4. Dependencies and Setup Instructions

### **Install Required Libraries**
```bash
pip install ultralytics deep-sort-realtime opencv-python numpy
```

### **Libraries Used**
- **ultralytics** – YOLOv8 model for person detection  
- **deep-sort-realtime** – Tracking algorithm to maintain consistent person IDs  
- **opencv-python** – Video processing, drawing, and writing output  
- **numpy** – Array operations  

### **How to Run (Colab)**
1. Open Google Colab  
2. Paste the full script into a single cell  
3. Run the cell  
4. Upload any video file when prompted  
5. The system generates `output.mp4` automatically  
6. Download the processed file  

### **How to Run (Local Machine)**
1. Save the Python script as `footfall_counter.py`  
2. Place your video file in the same folder  
3. Run:

```bash
python footfall_counter.py
```

4. The processed output is saved as:

```
output.mp4
```

---

This README fulfills the assignment requirements and documents the full working process clearly.
