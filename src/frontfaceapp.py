import csv
import datetime
import math
import sys
import cv2
import numpy as np
import os
import torch
os.environ['OMP_NUM_THREADS'] = '1'
import face_alignment
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#initialize face width variable
face_width = float(sys.argv[1])  # initialize face width variable
face_width_mm = face_width
#face_width_mm = 140.0
# Store the landmarks for the first face detected
first_face_landmarks = None
#initialize average point variable
avg_point = None
#headpose var
previous_head_pose = None
# lines will be shown by default
show_lines = True  
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

# Initialize the face alignment model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, flip_input=False)
print(device)
def draw_lines(frame):
    if show_lines:
        #Orange
        cv2.line(frame, point_34, point_9, (0, 125, 255), 1, cv2.LINE_AA)
        #Blue
        cv2.line(frame, avg_point, point_34, (255, 0, 0), 1, cv2.LINE_AA)
        #DarkGreen
        cv2.line(frame, point_22, point_23, (0, 100, 0), 1, cv2.LINE_AA)
        #Cyan
        cv2.line(frame, point_3, point_4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, point_4, point_5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, point_5, point_6, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, point_6, point_7, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, point_7, point_8, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(frame, point_8, point_9, (255, 255, 0), 1, cv2.LINE_AA)
        #Red
        cv2.line(frame, point_9, point_10, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(frame, point_10, point_11, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(frame, point_11, point_12, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(frame, point_12, point_13, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(frame, point_13, point_14, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(frame, point_14, point_15, (0, 0, 255), 1, cv2.LINE_AA)
        #Magenta
        cv2.line(frame, point_40, point_8, (255, 0, 255), 1, cv2.LINE_AA)
        #Purple
        cv2.line(frame, point_43, point_10, (0, 255, 125), 1, cv2.LINE_AA)
def on_checkbox_clicked(state):
    global show_lines
    show_lines = state

# Open a video capture stream from the default webcam
cap = cv2.VideoCapture('sample.mp4')

# create a window
#cv2.namedWindow("frame")

# create a checkbox and attach the callback function
#cv2.createButton("Show lines", on_checkbox_clicked)

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
video_path = os.path.join(timestamp_dir, f'front_recorded_{date_string}.mp4')
out = cv2.VideoWriter(video_path, fourcc, 30.0, (v_width, v_height)) #remember to change back to 30
# Initialize the CSV files
#landmark_csv_path = os.path.join(timestamp_dir, f'landmark_points_{date_string}.csv')
#distance_csv_path = os.path.join(timestamp_dir, f'eucledian_distances_{date_string}.csv')
#with open(landmark_csv_path, mode='w') as landmark_file, \
#        open(distance_csv_path, mode='w') as distance_file:
#    landmark_writer = csv.writer(landmark_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#    landmark_writer.writerow(['Frame No', 'Landmark No', 'X', 'Y'])
#    distance_writer = csv.writer(distance_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#    distance_writer.writerow(['Frame No', 'Distance(4,8)', 'Distance(5,9)', 'Distance(6,9)',
#                               'Distance(58,9)', 'Distance(7,11)', 'Distance(14,10)','Distance(13,9)',
#                               'Distance(12,9)'])
#Images Dir
images_dir = os.path.join(timestamp_dir, "front_images")
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
    (x,y) = map(int,faces[0][1-1])
    point_0 = (x,y)
    (x,y) = map(int,faces[0][2-1])
    point_2 = (x,y)
    (x,y) = map(int,faces[0][3-1])
    point_3 = (x,y)
    (x,y) = map(int,faces[0][4-1])
    point_4 = (x,y)
    (x,y) = map(int,faces[0][5-1])
    point_5 = (x,y)
    (x,y) = map(int,faces[0][6-1])
    point_6 = (x,y)
    (x,y) = map(int,faces[0][7-1])
    point_7 = (x,y)
    (x,y) = map(int,faces[0][8-1])
    point_8 = (x,y)
    (x,y) = map(int,faces[0][9-1])
    point_9 = (x,y)
    (x,y) = map(int,faces[0][10-1])
    point_10 = (x,y)
    (x,y) = map(int,faces[0][11-1])
    point_11 = (x,y)
    (x,y) = map(int,faces[0][12-1])
    point_12 = (x,y)
    (x,y) = map(int,faces[0][13-1])
    point_13 = (x,y)
    (x,y) = map(int,faces[0][14-1])
    point_14 = (x,y)
    (x,y) = map(int,faces[0][15-1])
    point_15 = (x,y)
    (x,y) = map(int,faces[0][17-1])
    point_17 = (x,y)
    (x,y) = map(int,faces[0][22-1])
    point_22 = (x,y)
    (x,y) = map(int,faces[0][23-1])
    point_23 = (x,y)
    (x,y) = map(int,faces[0][34-1])
    point_34 = (x,y)
    (x,y) = map(int,faces[0][40-1])
    point_40 = (x,y)
    (x,y) = map(int,faces[0][43-1])
    point_43 = (x,y)
    
    # draw the lines on the frame
    draw_lines(frame)
    
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
                cv2.circle(frame, (avg_point[0],avg_point[1]), 3, (0, 100, 0), -1)


        d1 = math.sqrt((avg_point[0] - point_34[0]) ** 2 +
                        (avg_point[1] - point_34[1]) ** 2)
        d2 = math.sqrt((point_9[0] - point_34[0]) ** 2 +
                        (point_9[1] - point_34[1]) ** 2)
        d3 = math.sqrt((point_40[0] - point_8[0]) ** 2 +
                        (point_40[1] - point_8[1]) ** 2)
        d4 = math.sqrt((point_43[0] - point_10[0]) ** 2 +
                        (point_43[1] - point_10[1]) ** 2)
        d_3_4 = math.sqrt((point_3[0] - point_4[0]) ** 2 +
                        (point_3[1] - point_4[1]) ** 2)
        d_4_5 = math.sqrt((point_5[0] - point_4[0]) ** 2 +
                        (point_5[1] - point_4[1]) ** 2)
        d_5_6 = math.sqrt((point_5[0] - point_6[0]) ** 2 +
                        (point_5[1] - point_6[1]) ** 2)
        d_6_7 = math.sqrt((point_6[0] - point_7[0]) ** 2 +
                        (point_6[1] - point_7[1]) ** 2)
        d_7_8 = math.sqrt((point_7[0] - point_8[0]) ** 2 +
                        (point_7[1] - point_8[1]) ** 2)
        d_8_9 = math.sqrt((point_8[0] - point_9[0]) ** 2 +
                        (point_8[1] - point_9[1]) ** 2)
        
        d_9_10 = math.sqrt((point_10[0] - point_9[0]) ** 2 +
                        (point_10[1] - point_9[1]) ** 2)
        d_10_11 = math.sqrt((point_10[0] - point_11[0]) ** 2 +
                        (point_10[1] - point_11[1]) ** 2)
        d_11_12 = math.sqrt((point_11[0] - point_12[0]) ** 2 +
                        (point_11[1] - point_12[1]) ** 2)
        d_12_13 = math.sqrt((point_12[0] - point_13[0]) ** 2 +
                        (point_12[1] - point_13[1]) ** 2)
        d_13_14 = math.sqrt((point_13[0] - point_14[0]) ** 2 +
                        (point_13[1] - point_14[1]) ** 2)
        d_14_15 = math.sqrt((point_14[0] - point_15[0]) ** 2 +
                        (point_14[1] - point_15[1]) ** 2)
        
        # Convert the distances from pixels to cm 
        x16, y16 = map(int, faces[0][16])
        x0, y0 = map(int, faces[0][0])
        face_width_px = x16 - x0
        #face_width_mm from input
        d1_mm = (d1 / face_width_px) * face_width_mm
        d2_mm = (d2 / face_width_px) * face_width_mm
        d3_mm = (d3 / face_width_px) * face_width_mm
        d4_mm = (d4 / face_width_px) * face_width_mm
        d_3_4_mm = (d_3_4 / face_width_px) * face_width_mm
        d_4_5_mm = (d_4_5 / face_width_px) * face_width_mm
        d_5_6_mm = (d_5_6 / face_width_px) * face_width_mm
        d_6_7_mm = (d_6_7 / face_width_px) * face_width_mm
        d_7_8_mm = (d_7_8 / face_width_px) * face_width_mm
        d_8_9_mm = (d_8_9 / face_width_px) * face_width_mm
        d_3_9_sum = d_3_4_mm + d_4_5_mm + d_5_6_mm + d_6_7_mm + d_7_8_mm + d_8_9_mm

        d_9_10mm = (d_9_10 / face_width_px) * face_width_mm
        d_10_11mm = (d_10_11 / face_width_px) * face_width_mm
        d_11_12mm = (d_11_12 / face_width_px) * face_width_mm
        d_12_13mm = (d_12_13 / face_width_px) * face_width_mm
        d_13_14mm = (d_13_14 / face_width_px) * face_width_mm
        d_14_15mm = (d_14_15 / face_width_px) * face_width_mm
        d_9_15_sum = d_9_10mm + d_10_11mm + d_11_12mm + d_12_13mm + d_13_14mm + d_14_15mm
        

        # Display the eucledian distances on the top right side of the frame
        text0_label="sZy left-sZy right:"
        text0=f"{face_width_mm:.2f}mm"

        text1=f"{d1_mm:.2f}mm"
        text1="sN-Sn:"+ text1
        text1=str(text1)

        text2_label = "Sn-sPog/Gn:"
        text2_label = str(text2_label)
        text2=f"{d2_mm:.2f}mm"
        
        text3_label = "iC line right:"
        text3=f"{d3_mm:.2f}mm"
        text3=text3_label+text3
        text3 = str(text3)

        text4_label = "iC line left:"
        text4=f"{d4_mm:.2f}mm"
        text4=text4_label+text4
        text4 = str(text4)
        
        text5_label= "Ar/Go-sPog/Gn right:"
        text5= f"{d_3_9_sum:.2f}mm"

        text6_label= "Ar/Go-sPog/Gn left:"
        text6= f"{d_9_15_sum:.2f}mm"

        text_width, text_height = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x_ec, y_ec = frame.shape[1] - text_width - 10, 30
        cv2.putText(frame, text0_label, (x_ec, y_ec),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text0, (x_ec, y_ec+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text1, (x_ec, y_ec+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text2_label, (x_ec, y_ec+80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text2, (x_ec, y_ec+100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text3, (x_ec, y_ec+120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text4, (x_ec, y_ec+140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 125), 1, cv2.LINE_AA)
        cv2.putText(frame, text5_label, (x_ec, y_ec+160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text5, (x_ec, y_ec+180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, text6_label, (x_ec, y_ec+200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, text6, (x_ec, y_ec+220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        

    end_time = datetime.datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"Frame: {frame_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    #Drawing Lines
    cv2.line(frame, point_0, point_17, (255, 255, 255), 2)
    # Display the frame
    cv2.namedWindow("Front Face Landmark Detection", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Front Face Landmark Detection", 900, 100)
    cv2.imshow('Front Face Landmark Detection', frame)
    # Save a snapshot of the GUI as an image
    cv2.imwrite(f"{images_dir}/{frame_count}.jpg", frame)
    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture stream and close all windows
cap.release()
out.release()

cv2.destroyAllWindows()
