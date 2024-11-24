import cv2
import pytesseract
import os

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
def main():
    # Open video or camera feed
    video = cv2.VideoCapture('video.mp4')  # Replace with 0 for webcam or specify your video file

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

if __name__ == "__main__":
    main()
