import cv2
import pytesseract
import os
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk

# Path to Tesseract executable (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path if needed

# Load Haar Cascades
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')  # Path to car cascade file
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')  # Path to license plate cascade file

# Create output directory for plates
if not os.path.exists("plates"):
    os.makedirs("plates")

# Function to detect license plates and perform OCR
def detect_and_save_license_plate(frame, car_region, frame_count):
    x, y, w, h = car_region
    car_roi = frame[y:y + h, x:x + w]  # Crop car region

    # Convert to grayscale
    gray_car = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)

    # Detect license plates within the car region
    plates = plate_cascade.detectMultiScale(gray_car, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (px, py, pw, ph) in enumerate(plates):
        # Crop license plate region
        plate_roi = car_roi[py:py + ph, px:px + pw]

        # Preprocess for better OCR
        plate_gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.threshold(plate_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Perform OCR
        plate_text = pytesseract.image_to_string(plate_gray, config='--psm 8')  # PSM 8 for single line of text
        plate_text = plate_text.strip()

        if plate_text:
            print(f"Detected Plate: {plate_text}")

            # Save plate text to file
            with open("detected_plates.txt", "a") as file:
                file.write(f"{plate_text}\n")

            # Save plate region as an image
            plate_image_path = f"plates/plate_{frame_count}_{i}.jpg"
            cv2.imwrite(plate_image_path, plate_roi)
            print(f"Saved plate image to {plate_image_path}")

            # Draw the plate text on the frame above the license plate
            cv2.putText(frame, plate_text, (x + px, y + py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw rectangle around the plate on the original frame
        cv2.rectangle(frame, (x + px, y + py), (x + px + pw, y + py + ph), (255, 0, 0), 2)

# Main function to process video or camera feed
def process_video(video_path):
    # Open video file
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video or stream

        frame_count += 1

        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

        for car in cars:
            x, y, w, h = car

            # Draw rectangle around detected car
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Detect and save license plates in the car region
            detect_and_save_license_plate(frame, car, frame_count)

        # Display the frame with annotations
        cv2.imshow("Car and License Plate Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

# Function to handle drag and drop event
def on_drop(event):
    video_path = event.data  # Get the dropped file path
    print(f"Video file dropped: {video_path}")
    process_video(video_path)  # Start video processing

# Function to create a gradient background using PIL
def create_gradient_background():
    width, height = 500, 300
    gradient = Image.new('RGB', (width, height), color=(255, 255, 255))

    for i in range(height):
        r = int(255 * (1 - i / height))
        g = int(255 * (i / height))
        b = 255
        for j in range(width):
            gradient.putpixel((j, i), (r, g, b))

    return gradient

# Function to load the cat image
def load_cat_image():
    cat_image = Image.open('cat.jpg')  # Make sure to have a "cat.jpg" image in the same directory
    cat_image = cat_image.resize((100, 100))  # Resize the image to fit well in the window
    return cat_image

# Tkinter window setup
def create_gui():
    root = TkinterDnD.Tk()  # Initialize TkinterDnD window
    root.title("Car and License Plate Detection")
    root.geometry("500x300")

    # Create gradient background
    gradient = create_gradient_background()
    gradient_img = ImageTk.PhotoImage(gradient)
    background_label = tk.Label(root, image=gradient_img)
    background_label.image = gradient_img  # Keep reference to avoid garbage collection
    background_label.place(relwidth=1, relheight=1)  # Stretch to fill window

    # Load and display the cat image
    cat_image = load_cat_image()
    cat_img_tk = ImageTk.PhotoImage(cat_image)
    cat_label = tk.Label(root, image=cat_img_tk)
    cat_label.image = cat_img_tk  # Keep reference
    cat_label.place(x=50, y=150)

    # Add the text "I am a graphic designer" with bold and italic font
    text_label = tk.Label(root, text="I am a graphic designer", font=("Arial", 14, "bold italic"), bg='white')
    text_label.place(x=150, y=50)

    # Add a label for instructions
    label = tk.Label(root, text="Drag and drop a video file here", font=("Arial", 14), bg='white')
    label.pack(pady=100)

    # Bind the drop event
    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', on_drop)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
