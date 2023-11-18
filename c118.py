import cv2

# Load the Haar Cascade Classifier for full body detection
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Read the video file
cap = cv2.VideoCapture("F:\1aa\coding\python projects\c118\walking.mp4")
 # Replace 'path_to_your_video_file.mp4' with the actual video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the frame
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)

    # Draw rectangles around detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the frame with detected bodies
    cv2.imshow('Detecting Bodies', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
