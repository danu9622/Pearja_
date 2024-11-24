import cv2
import pytesseract

# Path to Tesseract executable (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path as needed

# Load Haar Cascades for car and license plate detection
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')  # Replace with your car cascade file
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Function to perform license plate detection and OCR
def detect_license_plate(frame, car_region):
    x, y, w, h = car_region
    # Extract car region from the frame
    car_roi = frame[y:y + h, x:x + w]

    # Convert to grayscale
    gray_car = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)

    # Detect license plates within the car region
    plates = plate_cascade.detectMultiScale(gray_car, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (px, py, pw, ph) in plates:
        # Draw a rectangle around the detected plate
        cv2.rectangle(car_roi, (px, py), (px + pw, py + ph), (255, 0, 0), 2)

        # Extract the license plate region
        plate_region = car_roi[py:py + ph, px:px + pw]

        # Preprocess the plate region for OCR
        plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.threshold(plate_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Perform OCR to extract text
        plate_text = pytesseract.image_to_string(plate_gray, config='--psm 8')
        print(f"Detected Plate: {plate_text.strip()}")

        # Save the detected plate number to a text file
        with open("detected_plates.txt", "a") as file:
            file.write(plate_text.strip() + "\n")

        # Show the license plate text on the frame
        cv2.putText(frame, plate_text.strip(), (x + px, y + py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Main function to process video
def main():
    # Open video file (or use 0 for webcam)
    video = cv2.VideoCapture('video.mp4')  # Replace with your video file

    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video

        # Resize the frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Convert to grayscale for car detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

        for car in cars:
            x, y, w, h = car

            # Draw a rectangle around each detected car
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Call license plate detection within the car region
            detect_license_plate(frame, car)

        # Display the video with rectangles and OCR results
        cv2.imshow("Car and License Plate Detection", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
