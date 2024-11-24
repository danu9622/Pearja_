import cv2  # Import OpenCV library

# Load the Haar Cascade for car detection
car_cascade = cv2.CascadeClassifier('cars.xml')  # Make sure the file is in the same folder

# Open the video file
video = cv2.VideoCapture('video.mp4')  # Replace with your video file name

frame_count = 0  # Initialize a frame counter
cars_detected = []  # To store car positions

while True:
    # Read a frame from the video
    ret, frame = video.read()
    if not ret:
        break  # End of video

    frame_count += 1  # Increment frame counter

    # Only process every 5th frame for car detection
    if frame_count % 10 == 0:
        # Convert the frame to grayscale (required for Haar Cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars in the frame
        cars_detected = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Draw rectangles around the detected cars (use the last detection)
    for (x, y, w, h) in cars_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Resize the frame for better viewing
    frame_resized = cv2.resize(frame, (800, 600))  # Adjust the resolution as needed
    cv2.imshow('Optimized Car Detection', frame_resized)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
