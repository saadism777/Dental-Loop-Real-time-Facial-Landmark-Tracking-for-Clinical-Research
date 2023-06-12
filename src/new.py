import dlib
import cv2
import numpy as np

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\Projects\Dental-Loop-Real-time-Facial-Landmark-Tracking-for-Clinical-Research\models\shape_predictor_68_face_landmarks.dat")  # Replace with the path to your predictor file

# Initialize the tracker variables
prev_landmarks = None
smoothing_factor = 0.5  # Adjust this value to control the amount of smoothing (0.0 - no smoothing, 1.0 - full smoothing)

# Create a VideoCapture object to read from the webcam (change index to 0 if using the default camera)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for dlib processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    if len(faces) > 0:
        # Get the bounding box of the first face
        face = faces[0]

        # Determine the facial landmarks for the face region
        landmarks = predictor(gray, face)

        if prev_landmarks is not None:
            # Convert the landmarks to a NumPy array
            landmarks_np = np.array([[point.x, point.y] for point in landmarks.parts()])

            # Smooth the landmarks using a moving average filter
            smoothed_landmarks = smoothing_factor * landmarks_np + (1 - smoothing_factor) * prev_landmarks

            # Create a new dlib shape object with smoothed landmarks
            smoothed_landmarks_shape = dlib.full_object_detection(landmarks.rect, dlib.points(smoothed_landmarks.flatten()))

            # Draw the smoothed landmarks on the frame
            for point in smoothed_landmarks_shape.parts():
                cv2.circle(frame, (point.x, point.y), 1, (0, 0, 255), -1)

            # Update the previous landmarks with smoothed landmarks
            prev_landmarks = smoothed_landmarks

        else:
            # Draw the initial landmarks on the frame
            for point in landmarks.parts():
                cv2.circle(frame, (point.x, point.y), 1, (0, 0, 255), -1)

            # Initialize the previous landmarks
            prev_landmarks = np.array([[point.x, point.y] for point in landmarks.parts()])

    cv2.imshow("Facial Landmarks", frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
