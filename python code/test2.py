import cv2
import pytesseract
from ultralytics import YOLO
import os
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# YOLOv5 Model for car and plate detection
yolo_model = YOLO("yolov5n.pt")  # YOLOv5 Nano model (downloaded automatically)

# Create output directory for plates
if not os.path.exists("plates"):
    os.makedirs("plates")

# Function to detect license plates and perform OCR
def detect_and_save_license_plate(frame, results, frame_count):
    for result in results:
        for box, class_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if int(class_id) == 2:  # Class ID for "car" (adjust if using a custom model)
                x1, y1, x2, y2 = map(int, box)
                car_roi = frame[y1:y2, x1:x2]  # Crop car region

                # Detect license plates within the car region
                # Placeholder: Assume plate is detected (modify with proper plate detection logic if needed)
                plate_roi = car_roi  

                # Preprocess for OCR
                plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                plate_gray = cv2.threshold(plate_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

                # Perform OCR
                plate_text = pytesseract.image_to_string(plate_gray, config='--psm 8').strip()
                if plate_text:
                    print(f"Detected Plate: {plate_text}")

                    # Save detected text
                    with open("detected_plates.txt", "a") as file:
                        file.write(f"{plate_text}\n")

                    # Save plate region as an image
                    plate_path = f"plates/plate_{frame_count}.jpg"
                    cv2.imwrite(plate_path, plate_roi)
                    print(f"Saved plate image to {plate_path}")

                    # Draw detected text
                    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw rectangle around the car
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Function to process video
def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1

        # Resize for faster processing
        frame_resized = cv2.resize(frame, (640, 480))

        # YOLO detection
        results = yolo_model(frame_resized, conf=0.5)

        # Detect and annotate license plates
        detect_and_save_license_plate(frame_resized, results, frame_count)

        # Display the frame
        cv2.imshow("License Plate Detection", frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Function to handle drag-and-drop
def on_drop(event):
    video_path = event.data.strip('{}')  # Handle TkinterDnD formatting quirks
    print(f"Video dropped: {video_path}")
    process_video(video_path)

# Create GUI
def create_gui():
    root = TkinterDnD.Tk()
    root.title("License Plate Detection")
    root.geometry("500x300")

    label = tk.Label(root, text="Drag and drop a video file here", font=("Arial", 14))
    label.pack(pady=100)

    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', on_drop)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
