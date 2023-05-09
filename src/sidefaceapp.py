import datetime
import math
import sys
import cv2
import numpy as np
import os
import torch
os.environ['OMP_NUM_THREADS'] = '1'
import face_alignment

#initialize face width variable
face_width = float(sys.argv[1])  # initialize face width variable
face_width_mm = face_width

# Store the landmarks for the first face detected
first_face_landmarks = None
#initialize average point variable
avg_point = None
# Initialize the face alignment model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)
print(device)
# Get the path to the project's root directory
project_root = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the outputs directory in the project's root directory
output_dir = os.path.join(project_root, '..' , 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#Timestamp Dir
date_string = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M%p")
timestamp_dir = os.path.join(output_dir, date_string)
if not os.path.exists(timestamp_dir):
    os.makedirs(timestamp_dir)
# Open a video capture stream from the default webcam
cap = cv2.VideoCapture('sampleside.mp4')
#Defining resolution
#v_width = 640
#v_height = 480
v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'{v_height}+' '+{v_weight}')
cap.set(3, v_width)
cap.set(4, v_height)
# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = os.path.join(timestamp_dir, f'side_recorded_{date_string}.mp4')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (v_width, v_height)) #remember to change back to 30
#Images Dir
images_dir = os.path.join(timestamp_dir, "side_images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
# Initialize the variables
frame_count = 0
start_time = datetime.datetime.now()
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break
    # Increment the frame count
    frame_count += 1
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect facial landmarks using the FAN model
    faces = model.get_landmarks_from_image(gray)
    
    print(len(faces[0]))

    #Coordinate points for the landmarks
    (x,y) = map(int,faces[0][34-1])
    point_34 = (x,y)
    (x,y) = map(int,faces[0][9-1])
    point_9 = (x,y)
    (x,y) = map(int,faces[0][5-1])
    point_5 = (x,y)
    (x,y) = map(int,faces[0][2-1])
    point_2 = (x,y)

    #Orange
    cv2.line(frame, point_34, point_9, (0, 125, 255), 1, cv2.LINE_AA)
    #Blue
    cv2.line(frame, avg_point, point_34, (255, 0, 0), 1, cv2.LINE_AA)
    #Cyan
    cv2.line(frame, point_5, point_9, (255, 255, 0), 1, cv2.LINE_AA)
    #Magenta
    cv2.line(frame, point_2, point_9, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.line(frame, point_2, point_34, (255, 0, 255), 1, cv2.LINE_AA)
    if faces is not None:
        # If this is the first face detected, store its landmarks
        if first_face_landmarks is None:
            first_face_landmarks = faces[0]
        # If we have already detected the first face, use its landmarks to draw on subsequent frames
        else:
            faces[0] = first_face_landmarks
        # Draw the facial landmarks on the frame
        #landmark = faces[0]
        #x, y = map(int, landmark)
        for i in faces[0]:
                cv2.circle(frame, (int(i[0]), int(i[1])), 2, (0, 255, 0), -1)
                break
        
        # If landmark is 22 or 23, update avg_point
        if len(faces[0]) > 21 and len(faces[0]) > 22:
                x21, y21 = map(int, faces[0][21])
                x22, y22 = map(int, faces[0][22])
                avg_point = ((x21 + x22) // 2, (y21 + y22) // 2)
        # Display average point on frame
        if avg_point:
                cv2.circle(frame, (avg_point[0],avg_point[1]), 4, (0, 255, 0), -1)


        d1 = math.sqrt((avg_point[0] - point_34[0]) ** 2 +
                        (avg_point[1] - point_34[1]) ** 2)
        d2 = math.sqrt((point_9[0] - point_34[0]) ** 2 +
                        (point_9[1] - point_34[1]) ** 2)
        d3 = math.sqrt((point_9[0] - point_5[0]) ** 2 +
                        (point_9[1] - point_5[1]) ** 2)
        # Convert the distances from pixels to cm (assuming a face width of 15cm)
        x16, y16 = map(int, faces[0][16])
        x0, y0 = map(int, faces[0][0])
        face_width_px = x16 - x0
        #face_width_mm from input
        d1_mm = (d1 / face_width_px) * face_width_mm
        d2_mm = (d2 / face_width_px) * face_width_mm
        d3_mm = (d3 / face_width_px) * face_width_mm

        # Extract the landmarks of interest
        landmark_2 = faces[0][1]  # Landmark 2
        landmark_9 = faces[0][8]  # Landmark 9
        landmark_34 = faces[0][33]  # Landmark 34

        #Angle
        xA1,yA1 = landmark_9 - landmark_2
        xA2,yA2 = landmark_34 - landmark_2

        dot_product = xA1*xA2 + yA1*yA2
        length_1 = math.sqrt(xA1**2 + yA1**2)
        length_2 = math.sqrt(xA2**2 + yA2**2)
        angle = math.acos(dot_product / (length_1 * length_2))
        angle = math.degrees(angle)
        # Display the eucledian distances on the top right side of the frame
        text0_label="sZy left-sZy right:"
        text0=f"{face_width_mm:.2f}mm"

        text1=f"{d1_mm:.2f}mm"
        text1="sN-Sn:"+ text1
        text1=str(text1)

        text2_label = "Sn-sPog/Gn:"
        text2_label = str(text2_label)
        text2=f"{d2_mm:.2f}mm"

        text3_label = "Ar/Go-sPog/Gn right:"
        text3_label = str(text3_label)
        text3=f"{d3_mm:.2f}mm"
        text_width, text_height = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x_ec, y_ec = frame.shape[1] - text_width - 10, 30
        cv2.putText(frame, text0_label, (x_ec, y_ec),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text0, (x_ec, y_ec+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text1, (x_ec, y_ec+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text2_label, (x_ec, y_ec+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text2, (x_ec, y_ec+80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text3_label, (x_ec, y_ec+100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text3, (x_ec, y_ec+120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        # Display the angle on the frame
        cv2.putText(frame, f"Angle: {angle:.1f}",(x_ec, y_ec+140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"Frame: {frame_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    #Drawing Lines
    cv2.line(frame, (x16, y16), (x0, y0), (255, 255, 255), 2)
    # Display the frame
    cv2.namedWindow("Side Face Landmark Detection", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Side Face Landmark Detection", 0, 100)
    cv2.imshow('Side Face Landmark Detection', frame)
    # Save a snapshot of the GUI as an image
    cv2.imwrite(f"{images_dir}/{frame_count}.jpg", frame)
    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture stream and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
