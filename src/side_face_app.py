import csv
import datetime
import math
import os
import sys
import cv2
import matplotlib.pyplot as plt
import torch
import face_alignment
from PyQt5.QtWidgets import QDesktopWidget, QApplication, QWidget, QVBoxLayout, QCheckBox

# Open a video capture stream from the default webcam
cap = cv2.VideoCapture('..\sample\sample_side.mp4') # Enter 0 for webcam

# Initialize the variable to toggle the lines
show_lines = True 
show_lines_Ala_Tragus = True
show_lines_sN_Sn = True
show_lines_Sn_sPog = True
show_lines_Ar_sPog = True
show_lines_sN_Ar_Pog = True
# Checkbox Class for the lines
class Checkbox(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('SideFace: Show/Hide Lines')

        # Create the boolean variable
        self.boolean_variable = True

        # Create checkboxes
        checkbox0 = QCheckBox('All Lines:', self)
        checkbox0.setChecked(self.boolean_variable)
        checkbox0.stateChanged.connect(self.checkbox_state)

        checkbox1 = QCheckBox('Ala-Tragus :', self)
        checkbox1.setChecked(self.boolean_variable)
        checkbox1.stateChanged.connect(self.checkbox_state_Ala_Tragus)

        checkbox2 = QCheckBox('sN-Sn:', self)
        checkbox2.setChecked(self.boolean_variable)
        checkbox2.stateChanged.connect(self.checkbox_state_sN_Sn)

        checkbox3 = QCheckBox('Sn-sPog/Gn:', self)
        checkbox3.setChecked(self.boolean_variable)
        checkbox3.stateChanged.connect(self.checkbox_state_Sn_sPog)

        checkbox4 = QCheckBox('Ar/Go-sPog/Gn right:', self)
        checkbox4.setChecked(self.boolean_variable)
        checkbox4.stateChanged.connect(self.checkbox_state_Ar_sPog)

        checkbox5 = QCheckBox('Angle(sN-Ar-Pog):', self)
        checkbox5.setChecked(self.boolean_variable)
        checkbox5.stateChanged.connect(self.checkbox_state_sN_Ar_Pog)

        # Create a layout and add checkboxes to it
        layout = QVBoxLayout()
        layout.addWidget(checkbox0)
        layout.addWidget(checkbox1)
        layout.addWidget(checkbox2)
        layout.addWidget(checkbox3)
        layout.addWidget(checkbox4)
        layout.addWidget(checkbox5)

        # Set the layout for the main widget
        self.setLayout(layout)
        
        # Resize the window
        self.resize(300, 300)
        screen = QDesktopWidget().screenGeometry()

        # Calculate the center position of the screen
        center_x = screen.width() // 2
        center_y = screen.height() // 2

        # Move the window to the center of the screen
        self.move(center_x + 350 - self.width() // 2, center_y + 200 - self.height() // 2)
    def checkbox_state(self, state):
        sender = self.sender()
        global show_lines
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines = True
        else:
            show_lines = False

        print(f'{sender.text()} state changed: {show_lines}')
    
    def checkbox_state_Ala_Tragus(self, state):
        sender = self.sender()
        global show_lines_Ala_Tragus
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_Ala_Tragus = True
        else:
            show_lines_Ala_Tragus = False

        print(f'{sender.text()} state changed: {show_lines_Ala_Tragus}')
    
    def checkbox_state_sN_Sn(self, state):
        sender = self.sender()
        global show_lines_sN_Sn
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_sN_Sn = True
        else:
            show_lines_sN_Sn = False

        print(f'{sender.text()} state changed: {show_lines_sN_Sn}')
    
    def checkbox_state_Sn_sPog(self, state):
        sender = self.sender()
        global show_lines_Sn_sPog
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_Sn_sPog = True
        else:
            show_lines_Sn_sPog = False

        print(f'{sender.text()} state changed: {show_lines_Sn_sPog}')

    def checkbox_state_Ar_sPog(self, state):
        sender = self.sender()
        global show_lines_Ar_sPog
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_Ar_sPog = True
        else:
            show_lines_Ar_sPog = False

        print(f'{sender.text()} state changed: {show_lines_Ar_sPog}')

    def checkbox_state_sN_Ar_Pog(self, state):
        sender = self.sender()
        global show_lines_sN_Ar_Pog
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_sN_Ar_Pog = True
        else:
            show_lines_sN_Ar_Pog = False

        print(f'{sender.text()} state changed: {show_lines_sN_Ar_Pog}')

    # Cleaning up before quitting the app
    def closeEvent(self, event):
        # Generating graphs    
        graph(graph_1,'Soft tissue over nasion','Subnasale')
        graph(graph_2,'Subnasale','Soft tissue over Poginion')
        angle_graph(graph_3,'(sN-Ar','-Pog)')
        angle_graph(graph_4,'(sN-Sn','-Pog)')
        cap.release()
        out.release()
        event.accept()
        sys.exit(app.exec_())
        cv2.destroyAllWindows()
# initialize face width variable
face_width_mm = float(sys.argv[1])
#face_width_mm=84

# Store the landmarks for the first face detected
first_face_landmarks = None

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
# Initialize the CSV files
#landmark_csv_path = os.path.join(timestamp_dir, f'landmark_points_{date_string}.csv')
distance_csv_path = os.path.join(timestamp_dir, f'side_eucledian_distances_{date_string}.csv')
with open(distance_csv_path, mode='w') as distance_file:
    #landmark_writer = csv.writer(landmark_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #landmark_writer.writerow(['Frame No', 'Landmark No', 'X', 'Y'])
    distance_writer = csv.writer(distance_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    distance_writer.writerow(['Frame No','Time(s)', 'Ala-Tragus(mm)', 'sN-Sn(mm)', 'Sn-sPog/Gn(mm)',
                               'Ar/Go-sPog/Gn right(mm)', 'Angle(sN-Ar-Pog)(degree)', 'Angle(sN-Sn-Pog)(degree)'])
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
    output_file = f'{graph_path}/side_euclidean_distance_{p1}_{p2}.jpg'
    
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
        if show_lines_Sn_sPog:
                #Orange
                cv2.line(frame, point_34, point_9, (0, 125, 255), 1, cv2.LINE_AA)
        if show_lines_sN_Sn:
                if avg_point:
                        #Blue
                        cv2.line(frame, avg_point, point_34, (255, 0, 0), 1, cv2.LINE_AA)
        if show_lines_Ar_sPog:
                #Cyan
                cv2.line(frame, point_5, point_9, (255, 255, 0), 1, cv2.LINE_AA)
        if show_lines_sN_Ar_Pog:
                #Magenta
                cv2.line(frame, point_2, point_9, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.line(frame, point_2, point_34, (255, 0, 255), 1, cv2.LINE_AA)

# Function to calculate angles between facial lines
def calculate_angle(landmark_point_1, landmark_point_2, landmark_vertex):
    
    # Getting x,y coordinates 
    x1, y1 = landmark_point_1 - landmark_vertex
    x2, y2 = landmark_point_2 - landmark_vertex

    # Calculating Angle
    dot_product = x1 * x2 + y1 * y2
    length_1 = math.sqrt(x1 ** 2 + y1 ** 2)
    length_2 = math.sqrt(x2 ** 2 + y2 ** 2)
    angle = math.acos(dot_product / (length_1 * length_2))
    
    # Converting Radian to Degree
    angle = math.degrees(angle)
    
    return angle

# Initialize the variables
frame_count = 0
graph_1 = []
graph_2 = []
graph_3 = []
graph_4 = []
start_time = datetime.datetime.now()

# Starting the Checkbox app
app = QApplication(sys.argv)
window = Checkbox()
window.show()
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
        x33, y33 = map(int, faces[32])
        x2, y2 = map(int, faces[1])
        face_width_px = math.sqrt((x2 - x33) ** 2 +
                        (y2 - y33) ** 2)
        
        #face_width_mm from input
        d1_mm = (d1 / face_width_px) * face_width_mm
        d2_mm = (d2 / face_width_px) * face_width_mm
        d3_mm = (d3 / face_width_px) * face_width_mm

        # Extract the landmarks of interest
        landmark_2 = faces[2-1]  
        landmark_9 = faces[9-1]  
        landmark_34 = faces[34-1]  

        # Calling the Angle Function
        angle_1 = calculate_angle(landmark_9, landmark_34, landmark_2)
        angle_2 = calculate_angle(avg_point, landmark_9, landmark_34)
        
        graph_1.append((d1_mm))
        graph_2.append((d2_mm))
        graph_3.append((angle_1))
        graph_4.append((angle_2))
       
        #Display the eucledian distances on the top right side of the frame
        text0_label="Ala-Tragus:"
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
        text4 = f"{angle_1:.1f}"
        text5_label="Angle(sN-Sn-Pog):"
        text5 = f"{angle_2:.1f}"

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
    #Write the frame number and eucledian distances to the distance CSV file
    with open(distance_csv_path, mode='a') as distance_file:
        distance_writer = csv.writer(distance_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        distance_writer.writerow([frame_count,elapsed_time_real,face_width_mm, d1_mm, d2_mm, d3_mm, angle_1, angle_2])
    # Calculate the elapsed time
    elapsed_time_real = frame_count/ fps_real
    cv2.putText(frame, f"Frame: {frame_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Time: {elapsed_time_real:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    #Drawing Lines
    if show_lines_Ala_Tragus:
        cv2.line(frame, (x33, y33), (x2, y2), (255, 255, 255), 2)
    # Display the frame
    #cv2.namedWindow("Side Face Landmark Detection", cv2.WINDOW_NORMAL)
    #cv2.moveWindow("Side Face Landmark Detection", 0, 100)

    cv2.imshow('Side Face Landmark Detection', frame)
    out.write(frame)
    # Save a snapshot of the GUI as an image
    cv2.imwrite(f"{images_dir}/{frame_count}.jpg", frame)
    # Wait for a key press to exit
    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty("Side Face Landmark Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture stream and close all windows
graph(graph_1,'Soft tissue over nasion','Subnasale')
graph(graph_2,'Subnasale','Soft tissue over Poginion')
angle_graph(graph_3,'(sN-Ar','-Pog)')
angle_graph(graph_4,'(sN-Sn','-Pog)')
cap.release()
out.release()
cv2.destroyAllWindows()