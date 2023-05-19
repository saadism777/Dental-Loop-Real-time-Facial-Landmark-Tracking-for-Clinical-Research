import datetime
import math
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import face_alignment

# initialize face width variable
face_width_mm = float(sys.argv[1])
#face_width_mm=84

# Store the landmarks for the first face detected
first_face_landmarks = None

# Initialize the variable to toggle the lines
show_lines = True 

# Initialize average point variable
avg_point = None

# Initialize the face alignment model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector='sfd')
print(device)

# Get the path to the project's root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the outputs directory in the project's root directory
output_dir = os.path.join(project_root, '..' , 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Timestamp Dir
date_string = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M%p")
timestamp_dir = os.path.join(output_dir, date_string)
if not os.path.exists(timestamp_dir):
    os.makedirs(timestamp_dir)
    
# Open a video capture stream from the default webcam
cap = cv2.VideoCapture('Input video path') # Enter 0 for webcame
# Getting resolution of the input
v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Setting the resolution 
cap.set(3, v_width)
cap.set(4, v_height)
# Create a named window
cv2.namedWindow('Side Face Landmark Detection', cv2.WINDOW_NORMAL)

# Resize the window
cv2.resizeWindow('Side Face Landmark Detection', 640, 480)

# Get the screen width and height
screen_width, screen_height = 1920, 1080 # change this to your screen resolution

# Calculate the new position for the window
window_width, window_height = 640, 480
new_x = screen_width - window_width
new_y = 0

# Move the window to the new position
cv2.moveWindow('Side Face Landmark Detection', new_x, new_y)
#print(f'{v_height}+' '+{v_weight}')


# Get the frame rate
fps_real = cap.get(cv2.CAP_PROP_FPS)
elapsed_time_real=0

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = os.path.join(timestamp_dir, f'side_recorded_{date_string}.mp4')
out = cv2.VideoWriter(video_path, fourcc, fps_real, (v_width, v_height)) 

#Images Dir
images_dir = os.path.join(timestamp_dir, "side_images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Create graph image
graph_path = os.path.join(timestamp_dir, "graphs")
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

def graph(dist,p1,p2):
    # Create a list of frame numbers for the x-axis
    #frame_numbers = list(range(len(dist)))
    # Calculate the time array
    time_array = [i / fps_real for i in range(len(dist))]
    plt.ylim(40, 150)
    # Plot the distance values for the first distance in the array
    plt.plot(time_array, dist)
    
    # Add title and labels
    plt.title(f'Euclidean Distance of Landmarks {p1} and {p2}')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')
    output_file = f'{graph_path}/euclidean_distance_{p1}_{p2}.jpg'
    
    plt.savefig(output_file)

    # Save the plot as an image file
    plt.savefig(output_file)
    plt.clf()

def angle_graph(angle,p1,p2):
    # Create a list of frame numbers for the x-axis
    #frame_numbers = list(range(len(dist)))
    # Calculate the time array
    time_array = [i / fps_real for i in range(len(angle))]
    plt.ylim(0, 180)

    # Plot the distance values for the first distance in the array
    plt.plot(time_array, angle)
    
    # Add title and labels
    plt.title(f'Angle between {p1}{p2}')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degree)')
    output_file = f'{graph_path}/Angle_{p1}{p2}.jpg'
    
    plt.savefig(output_file)

    # Save the plot as an image file
    plt.savefig(output_file)
    plt.clf()

def draw_lines(frame):
    if show_lines:
        #Orange
        cv2.line(frame, point_34, point_9, (0, 125, 255), 1, cv2.LINE_AA)
        if avg_point:
                #Blue
                cv2.line(frame, avg_point, point_34, (255, 0, 0), 1, cv2.LINE_AA)
        #Cyan
        cv2.line(frame, point_5, point_9, (255, 255, 0), 1, cv2.LINE_AA)
        #Magenta
        cv2.line(frame, point_2, point_9, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.line(frame, point_2, point_34, (255, 0, 255), 1, cv2.LINE_AA)
        
# Initialize the variables
frame_count = 0
g1 = []
g2 = []
g3 = []
g4 = []
g5 = []
g6 = []
g7 = []
g8 = []
start_time = datetime.datetime.now()

# Process the video
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
    
    # define the size and position of the black box
    box_width = 150
    box_height = 50
   
    # get the dimensions of the frame
    height, width, _ = frame.shape
    box_x = width - box_width
    box_y = 0
   
    # draw the black box on the frame
    cv2.rectangle(frame, (box_x, box_y), (width, height), (0, 0, 0), -1)
    
    faces=faces[0]
    print(len(faces[0]))
    #Coordinate points for the landmarks  
    if len(faces) > 0:
        (x,y) = map(int,faces[34-1])
        point_34 = (x,y)
        (x,y) = map(int,faces[9-1])
        point_9 = (x,y)
        (x,y) = map(int,faces[5-1])
        point_5 = (x,y)
        (x,y) = map(int,faces[2-1])
        point_2 = (x,y)
       
        # draw the lines on the frame
        draw_lines(frame) 
        for landmark in faces:
            x, y = map(int, landmark)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # If landmark is 22 or 23, update avg_point
        if len(faces) > 21 and len(faces) > 22:
                x21, y21 = map(int, faces[21])
                x22, y22 = map(int, faces[22])
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
        x16, y16 = map(int, faces[32])
        x0, y0 = map(int, faces[1])
        face_width_px = math.sqrt((x0 - x16) ** 2 +
                        (y0 - y16) ** 2)
        
        #face_width_mm from input
        d1_mm = (d1 / face_width_px) * face_width_mm
        d2_mm = (d2 / face_width_px) * face_width_mm
        d3_mm = (d3 / face_width_px) * face_width_mm

        # Extract the landmarks of interest
        landmark_2 = faces[1]  
        landmark_9 = faces[8]  
        landmark_34 = faces[33]  

        #Angle
        xA1,yA1 = landmark_9 - landmark_2
        xA2,yA2 = landmark_34 - landmark_2

        dot_product = xA1*xA2 + yA1*yA2
        length_1 = math.sqrt(xA1**2 + yA1**2)
        length_2 = math.sqrt(xA2**2 + yA2**2)
        angle = math.acos(dot_product / (length_1 * length_2))
        angle = math.degrees(angle)

        xA3,yA3 = avg_point - landmark_34
        xA4,yA4 = landmark_34 - landmark_9

        dot_product2 = xA3*xA4 + yA3*yA4
        length_3 = math.sqrt(xA3**2 + yA3**2)
        length_2 = math.sqrt(xA4**2 + yA4**2)
        angle2 = math.acos(dot_product2 / (length_3 * length_2))
        angle2 = math.degrees(angle2)
        
        g1.append((d1_mm))
        g2.append((d2_mm))
        g3.append((angle))
        g4.append((angle2))
       
        #Display the eucledian distances on the top right side of the frame
        text0_label="Ala-Tragus :"
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
        
        text4_label="Angle(sN-Ar-Pog):"
        text4 = f"{angle:.1f}"
        text5_label="Angle(sN-Sn-Pog):"
        text5 = f"{angle2:.1f}"

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
        cv2.putText(frame, text4_label,(x_ec, y_ec+145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text4,(x_ec, y_ec+165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text5_label,(x_ec, y_ec+185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text5,(x_ec, y_ec+205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 255), 1, cv2.LINE_AA)
    end_time = datetime.datetime.now()
    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    fps = frame_count / elapsed_time
    
    # Calculate the elapsed time
    elapsed_time_real = frame_count/ fps_real
    cv2.putText(frame, f"Frame: {frame_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Time: {elapsed_time_real:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    #Drawing Lines
    cv2.line(frame, (x16, y16), (x0, y0), (255, 255, 255), 2)
    # Display the frame
    #cv2.namedWindow("Side Face Landmark Detection", cv2.WINDOW_NORMAL)
    #cv2.moveWindow("Side Face Landmark Detection", 0, 100)

    cv2.imshow('Side Face Landmark Detection', frame)
    out.write(frame)
    # Save a snapshot of the GUI as an image
    cv2.imwrite(f"{images_dir}/{frame_count}.jpg", frame)
    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture stream and close all windows
graph(g1,'Soft tissue over nasion','Subnasale')
graph(g2,'Subnasale','Soft tissue over Poginion')
angle_graph(g3,'(sN-Ar','-Pog)')
angle_graph(g4,'(sN-Sn','-Pog)')
cap.release()
out.release()
cv2.destroyAllWindows()