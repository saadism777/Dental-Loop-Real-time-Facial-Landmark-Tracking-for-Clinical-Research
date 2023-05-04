#prototype  dlib v3
import cv2
import dlib
import math
import datetime
import csv
import os
import matplotlib.pyplot as plt
import tkinter as tk
import threading
import numpy as np

face_width_mm = 0  # initialize face width variable

#Defining resolution
v_width = 640
v_height = 480

#function for setting face width
def set_face_width():
    global face_width_mm
    face_width_mm = float(face_width_entry.get())
    print(f"Face width set to {face_width_mm} mm")

root = tk.Tk()

# Define the callback function for the trackbar
def toggle_lines(state):
    global show_lines
    show_lines = state

# Create a named window for the display and add a trackbar
cv2.namedWindow('Settings')
#cv2.imshow("Face Landmarks", frame)
# Create a toggle button on the window
cv2.createTrackbar('Toggle Lines', 'Settings', 1, 1, toggle_lines)

# Initialize the variable to toggle the lines
show_lines = True    
# Add text box for entering face width
face_width_label = tk.Label(root, text="Enter face width (mm):")
face_width_label.pack()
face_width_entry = tk.Entry(root)
face_width_entry.pack()

# Add "Set Face Width" button
set_face_width_button = tk.Button(root, text="Set Face Width", command=set_face_width)
set_face_width_button.pack()

root.mainloop()

# Get the path to the project's root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the outputs directory in the project's root directory
output_dir = os.path.join(project_root, '..' , 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the path to the models directory
models_dir = os.path.join(project_root, '..', 'models')

# Load the shape predictor model from the models directory
shape_predictor_path = os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Initialize the webcam
cap = cv2.VideoCapture(0)

#Load the video file
#cap = cv2.VideoCapture('/Videos/sample.mp4')

cap.set(3, v_width)
cap.set(4, v_height)

#Timestamp Dir
date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
timestamp_dir = os.path.join(output_dir, date_string)
if not os.path.exists(timestamp_dir):
    os.makedirs(timestamp_dir)

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = os.path.join(timestamp_dir, f'recorded_{date_string}.mp4')
out = cv2.VideoWriter(video_path, fourcc, 10.0, (v_width, v_height)) #remember to change back to 30

# Initialize the CSV files
landmark_csv_path = os.path.join(timestamp_dir, f'landmark_points_{date_string}.csv')
distance_csv_path = os.path.join(timestamp_dir, f'eucledian_distances_{date_string}.csv')
with open(landmark_csv_path, mode='w') as landmark_file, \
        open(distance_csv_path, mode='w') as distance_file:
    landmark_writer = csv.writer(landmark_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    landmark_writer.writerow(['Frame No', 'Landmark No', 'X', 'Y'])
    distance_writer = csv.writer(distance_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    distance_writer.writerow(['Frame No', 'Distance(4,8)', 'Distance(5,9)', 'Distance(6,9)',
                               'Distance(58,9)', 'Distance(7,11)', 'Distance(14,10)','Distance(13,9)',
                               'Distance(12,9)'])
#Images Dir
images_dir = os.path.join(timestamp_dir, "images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Create graph image
graph_path = os.path.join(timestamp_dir, "graphs")
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

def graph(dist,p1,p2):
    # Create a list of frame numbers for the x-axis
    frame_numbers = list(range(len(dist)))
    plt.ylim(0, 100)
    # Plot the distance values for the first distance in the array
    plt.plot(frame_numbers, dist)
    
    # Add title and labels
    plt.title(f'Euclidean Distance of Landmarks {p1} and {p2}')
    plt.xlabel('Frame Number')
    plt.ylabel('Distance (mm)')
    output_file = f'{graph_path}/euclidean_distance_{p1}_{p2}.jpg'
    
    plt.savefig(output_file)
    # Save the plot as an image file
    plt.savefig(output_file)
    plt.clf()
# Initialize the variables
frame_count = 0
start_time = datetime.datetime.now()
g1 = []
g2 = []
g3 = []
g4 = []
g5 = []
g6 = []
g7 = []
g8 = []
first_face_position = None
# Start the loop to process each frame of the webcam feed
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Increment the frame count
    frame_count += 1
    
    # get the dimensions of the frame
    height, width, _ = frame.shape
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # define the size and position of the black box
    box_width = 100
    box_height = 50
    box_x = width - box_width
    box_y = 0
    # draw the black box on the frame
    cv2.rectangle(frame, (box_x, box_y), (width, height), (0, 0, 0), -1)

    # Detect faces using dlib's face detector
    faces = detector(gray, 0)

    # Only process the first face detected
    if len(faces) > 0:
        # Get the facial landmarks for the first face
        if first_face_position is None:
            first_face_position = faces[0]

        # Determine the distance from subsequent faces to the first face
        distances = [abs(face.left() - first_face_position.left()) +
                     abs(face.top() - first_face_position.top()) for face in faces]
        
        # Lock onto the face that is closest to the first face
        closest_face_index = distances.index(min(distances))
        face = faces[closest_face_index]
        landmarks = predictor(gray, face)
        
        # Draw line between landmarks 16 and 0 for facial width reference
        if show_lines:
            x1, y1 = landmarks.part(16).x, landmarks.part(16).y
            x2, y2 = landmarks.part(0).x, landmarks.part(0).y
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            #Drawing dotted lines for point pairs
            point_4 = (landmarks.part(4-1).x, landmarks.part(4-1).y)
            point_8 = (landmarks.part(8-1).x, landmarks.part(8-1).y)
            #Blue
            cv2.line(frame, point_4, point_8, (255, 0, 0), 1, cv2.LINE_AA)

            point_5 = (landmarks.part(5-1).x, landmarks.part(5-1).y)
            point_9 = (landmarks.part(9-1).x, landmarks.part(9-1).y)
            #Orange
            cv2.line(frame, point_5, point_9, (0, 125, 255), 1, cv2.LINE_AA)

            point_6 = (landmarks.part(6-1).x, landmarks.part(6-1).y)
            #Green
            cv2.line(frame, point_6, point_9, (0, 255, 125), 1, cv2.LINE_AA)

            point_58 = (landmarks.part(58-1).x, landmarks.part(58-1).y)
            #DarkGreen
            cv2.line(frame, point_58, point_9, (0, 100, 0), 1, cv2.LINE_AA)

            point_7 = (landmarks.part(7-1).x, landmarks.part(7-1).y)
            point_11 = (landmarks.part(11-1).x, landmarks.part(11-1).y)
            #Yellow
            cv2.line(frame, point_7, point_11, (0, 255, 255), 1, cv2.LINE_AA)

            point_14 = (landmarks.part(14-1).x, landmarks.part(14-1).y)
            point_10 = (landmarks.part(10-1).x, landmarks.part(10-1).y)
            #Cyan
            cv2.line(frame, point_14, point_10, (255, 255, 0), 1, cv2.LINE_AA)

            point_13 = (landmarks.part(13-1).x, landmarks.part(13-1).y)
            #Magenta
            cv2.line(frame, point_13, point_9, (255, 0, 255), 1, cv2.LINE_AA)
            point_12 = (landmarks.part(12-1).x, landmarks.part(12-1).y)
            #Red
            cv2.line(frame, point_12, point_9, (0, 0, 255), 1, cv2.LINE_AA)
        # Loop through each landmark point
        for i in range(68):
            # Get the x,y coordinates of the landmark point
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            # Draw a circle around the landmark point
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Write the landmark point coordinates to the landmark CSV file
            with open(landmark_csv_path, mode='a') as landmark_file:
                landmark_writer = csv.writer(landmark_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                landmark_writer.writerow([frame_count, i, x, y])

        # Calculate the eucledian distance between three specific landmark points
        d1 = math.sqrt((landmarks.part(4-1).x - landmarks.part(8-1).x) ** 2 +
                       (landmarks.part(4-1).y - landmarks.part(8-1).y) ** 2)
        d2 = math.sqrt((landmarks.part(5-1).x - landmarks.part(9-1).x) ** 2 +
                       (landmarks.part(5-1).y - landmarks.part(9-1).y) ** 2)
        d3 = math.sqrt((landmarks.part(6-1).x - landmarks.part(9-1).x) ** 2 +
                       (landmarks.part(6-1).y - landmarks.part(9-1).y) ** 2)
        d4 = math.sqrt((landmarks.part(58-1).x - landmarks.part(9-1).x) ** 2 +
                       (landmarks.part(58-1).y - landmarks.part(9-1).y) ** 2)
        d5 = math.sqrt((landmarks.part(7-1).x - landmarks.part(11-1).x) ** 2 +
                       (landmarks.part(7-1).y - landmarks.part(11-1).y) ** 2)
        d6 = math.sqrt((landmarks.part(14-1).x - landmarks.part(10-1).x) ** 2 +
                       (landmarks.part(14-1).y - landmarks.part(10-1).y) ** 2)
        d7 = math.sqrt((landmarks.part(13-1).x - landmarks.part(9-1).x) ** 2 +
                       (landmarks.part(13-1).y - landmarks.part(9-1).y) ** 2)
        d8 = math.sqrt((landmarks.part(12-1).x - landmarks.part(9-1).x) ** 2 +
                       (landmarks.part(12-1).y - landmarks.part(9-1).y) ** 2)
            # Convert the distances from pixels to cm (assuming a face width of 15cm)
        face_width_px = landmarks.part(16).x - landmarks.part(0).x
        #face_width_mm from input
        d1_mm = (d1 / face_width_px) * face_width_mm
        d2_mm = (d2 / face_width_px) * face_width_mm
        d3_mm = (d3 / face_width_px) * face_width_mm
        d4_mm = (d4 / face_width_px) * face_width_mm
        d5_mm = (d5 / face_width_px) * face_width_mm
        d6_mm = (d6 / face_width_px) * face_width_mm
        d7_mm = (d7 / face_width_px) * face_width_mm
        d8_mm = (d8 / face_width_px) * face_width_mm

        g1.append((d1_mm))
        g2.append((d2_mm))
        g3.append((d3_mm))
        g4.append((d4_mm))
        g5.append((d5_mm))
        g6.append((d6_mm))
        g7.append((d7_mm))
        g8.append((d8_mm))

        # Write the frame number and eucledian distances to the distance CSV file
        with open(distance_csv_path, mode='a') as distance_file:
            distance_writer = csv.writer(distance_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            distance_writer.writerow([frame_count, d1_mm, d2_mm, d3_mm, d4_mm, d5_mm, d6_mm, d7_mm, d8_mm])

        # Display the eucledian distances on the top right side of the frame
        text1=f"{d1_mm:.2f}mm"
        text2=f"{d2_mm:.2f}mm"
        text3= f"{d3_mm:.2f}mm"
        text4= f"{d4_mm:.2f}mm"
        text5= f"{d5_mm:.2f}mm"
        text6= f"{d6_mm:.2f}mm"
        text7= f"{d7_mm:.2f}mm"
        text8= f"{d8_mm:.2f}mm"
        text_width, text_height = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x_ec, y_ec = frame.shape[1] - text_width - 10, 30
        cv2.putText(frame, "Euclidean:-", (x_ec, y_ec),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text1, (x_ec, y_ec+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text2, (x_ec, y_ec+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text3, (x_ec, y_ec+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 125), 1, cv2.LINE_AA)
        cv2.putText(frame, text4, (x_ec, y_ec+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text5, (x_ec, y_ec+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text6, (x_ec, y_ec+120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text7, (x_ec, y_ec+140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text8, (x_ec, y_ec+160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Face width:-", (x_ec, y_ec+200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{face_width_mm}mm", (x_ec, y_ec+215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # If no faces are detected, reset the first_face_position variable
        first_face_position = None
    # Display the frame number and fps on the top left side of the frame
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"Frame: {frame_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    

    # Display the frame
    cv2.imshow("Facial Landmarks", frame)
    #cv2.createButton("Toggle Lines", toggle_lines, None, cv2.QT_CHECKBOX, False)
    
    out.write(frame)
    # Save a snapshot of the GUI as an image
    cv2.imwrite(f"{images_dir}/{frame_count}.jpg", frame)
    graph(g1,4,8)
    graph(g2,5,9)
    graph(g3,6,9)
    graph(g4,58,9)
    graph(g5,7,11)
    graph(g6,14,10)
    graph(g7,13,9)
    graph(g8,12,9)

    
    # Wait for a key press to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

    
cap.release()
out.release()

cv2.destroyAllWindows()