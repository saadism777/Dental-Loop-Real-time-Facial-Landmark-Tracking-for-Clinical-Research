#prototype  dlib v3
import csv
import datetime
import math
import os
import sys
import time
import cv2
import dlib
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout,QCheckBox, QDesktopWidget, QPushButton

# Initialize the webcam or video file path
#path = '..\sample\sample_front.mp4' #video file path or 0 for webcam
#path = 1
path = str(sys.argv[2])
if path=='None':
    path = 0

cap = cv2.VideoCapture(path)

if isinstance(path, int):
    # Set the desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# initialize face width variable
face_width_mm = float(sys.argv[1])  
#face_width_mm = 152
print(sys.argv[2])
# Initialize the variables to toggle the lines
show_lines = True    
show_lines_sZy = True
show_lines_sN_Sn = True
show_lines_Sn_sPog = True
show_lines_iC_Left = True
show_lines_iC_Right = True
show_lines_Ar_Left = True
show_lines_Ar_Right = True
recording_status = False
# Checkbox Class for the lines
class Checkbox(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('FrontFace: Show/Hide Lines')

        # Create the boolean variable
        self.boolean_variable = True

        # Create checkboxes
        checkbox0 = QCheckBox('All Lines:', self)
        checkbox0.setChecked(self.boolean_variable)
        checkbox0.stateChanged.connect(self.checkbox_state)

        checkbox1 = QCheckBox('sZy left-sZy right:', self)
        checkbox1.setChecked(self.boolean_variable)
        checkbox1.stateChanged.connect(self.checkbox_state_sZy)

        checkbox2 = QCheckBox('sN-Sn:', self)
        checkbox2.setChecked(self.boolean_variable)
        checkbox2.stateChanged.connect(self.checkbox_state_sN_Sn)

        checkbox3 = QCheckBox('Sn-sPog/Gn:', self)
        checkbox3.setChecked(self.boolean_variable)
        checkbox3.stateChanged.connect(self.checkbox_state_Sn_sPog)

        checkbox4 = QCheckBox('iC lines left:', self)
        checkbox4.setChecked(self.boolean_variable)
        checkbox4.stateChanged.connect(self.checkbox_state_iC_Left)

        checkbox5 = QCheckBox('iC lines right:', self)
        checkbox5.setChecked(self.boolean_variable)
        checkbox5.stateChanged.connect(self.checkbox_state_iC_Right)

        checkbox6 = QCheckBox('Ar/Go-sPog/Gn left:', self)
        checkbox6.setChecked(self.boolean_variable)
        checkbox6.stateChanged.connect(self.checkbox_state_Ar_Left)

        checkbox7 = QCheckBox('Ar/Go-sPog/Gn right:', self)
        checkbox7.setChecked(self.boolean_variable)
        checkbox7.stateChanged.connect(self.checkbox_state_Ar_Right)
        
        if isinstance(path, int):
            start_btn = QPushButton('Start Recording', self)
            stop_btn = QPushButton('Stop Recording', self)
        
            start_btn.clicked.connect(self.start_recording)
            stop_btn.clicked.connect(self.stop_recording)

        # Create a layout and add checkboxes to it
        layout = QVBoxLayout()
        if isinstance(path, int):
            layout.addWidget(start_btn)
            layout.addWidget(stop_btn)

        layout.addWidget(checkbox0)
        layout.addWidget(checkbox1)
        layout.addWidget(checkbox2)
        layout.addWidget(checkbox3)
        layout.addWidget(checkbox4)
        layout.addWidget(checkbox5)
        layout.addWidget(checkbox6)
        layout.addWidget(checkbox7)

        # Set the layout for the main widget
        self.setLayout(layout)

        # Resize the window
        self.resize(300, 300)

        screen = QDesktopWidget().screenGeometry()

        # Calculate the center position of the screen
        center_x = screen.width() // 2
        center_y = screen.height() // 2

        # Move the window to the center of the screen
        self.move(center_x - 350 - self.width() // 2, center_y + 200 - self.height() // 2)
    
    def start_recording(self):
        global recording_status
        recording_status = True
        print("Recording started")

    def stop_recording(self):
        global recording_status
        recording_status = False
        print("Recording stopped")

    def checkbox_state(self, state):
        sender = self.sender()
        global show_lines
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines = True
        else:
            show_lines = False

        print(f'{sender.text()} state changed: {show_lines}')
    
    def checkbox_state_sZy(self, state):
        sender = self.sender()
        global show_lines_sZy
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_sZy = True
        else:
            show_lines_sZy = False

        print(f'{sender.text()} state changed: {show_lines_sZy}')
    
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

    def checkbox_state_iC_Left(self, state):
        sender = self.sender()
        global show_lines_iC_Left
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_iC_Left = True
        else:
            show_lines_iC_Left = False

        print(f'{sender.text()} state changed: {show_lines_iC_Left}')
    
    def checkbox_state_iC_Right(self, state):
        sender = self.sender()
        global show_lines_iC_Right
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_iC_Right = True
        else:
            show_lines_iC_Right = False

        print(f'{sender.text()} state changed: {show_lines_iC_Right}')
    
    def checkbox_state_Ar_Left(self, state):
        sender = self.sender()
        global show_lines_Ar_Left
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_Ar_Left = True
        else:
            show_lines_Ar_Left = False

        print(f'{sender.text()} state changed: {show_lines_iC_Left}')

    def checkbox_state_Ar_Right(self, state):
        sender = self.sender()
        global show_lines_Ar_Right
        # Update the boolean variable based on checkbox state
        if state == 2:  # Qt.Checked
            show_lines_Ar_Right = True
        else:
            show_lines_Ar_Right = False

        print(f'{sender.text()} state changed: {show_lines_Ar_Right}')
    
    # Cleaning up before quitting the app
    def closeEvent(self, event):
        # Generating graphs    
        graph(graph_1,'Soft tissue over nasion','Subnasale')
        graph(graph_2,'Subnasale','Soft tissue over Poginion')
        out.release()   
        cap.release()
        event.accept()
        sys.exit(app.exec_())
        cv2.destroyAllWindows()
        


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

# Get the path to the models directory
models_dir = os.path.join(project_root, '..', 'models')

# Load the shape predictor model from the models directory
shape_predictor_path = os.path.join(models_dir,'shape_predictor_68_face_landmarks.dat')

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Get the frame rate and resolution
fps_real = cap.get(cv2.CAP_PROP_FPS)
v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setting the OpenCV frame resolution
cap.set(3, v_width)
cap.set(4, v_height)

# Create a named window
cv2.namedWindow('Front Face Landmark Detection', cv2.WINDOW_NORMAL)

# Resize the window
cv2.resizeWindow('Front Face Landmark Detection', 640, 480)

# Get the screen width and height
screen_width, screen_height = 1920, 1080 # Change these values to match your screen size

# Calculate the x and y positions to center the window
x = 0
y = 0

# Position the window at the left center of the screen
cv2.moveWindow('Front Face Landmark Detection', x, y)

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = os.path.join(timestamp_dir, f'front_recorded_{date_string}.mp4')
out = cv2.VideoWriter(video_path, fourcc, fps_real, (v_width, v_height)) #remember to change back to 30

# Initialize the CSV files
#landmark_csv_path = os.path.join(timestamp_dir, f'landmark_points_{date_string}.csv')
distance_csv_path = os.path.join(timestamp_dir, f'front_eucledian_distances_{date_string}.csv')
with open(distance_csv_path, mode='w') as distance_file:
    #landmark_writer = csv.writer(landmark_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #landmark_writer.writerow(['Frame No', 'Landmark No', 'X', 'Y'])
    distance_writer = csv.writer(distance_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    distance_writer.writerow(['Frame No','Time(s)', 'sZy left-sZy right(mm)', 'sN-Sn(mm)', 'Sn-sPog/Gn(mm)',
                               'iC line right(mm)', 'iC line left(mm)', 'Ar/Go-sPog/Gn right(mm)','Ar/Go-sPog/Gn left(mm)'])
#Images Dir
images_dir = os.path.join(timestamp_dir, "front images")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Create graph image path
graph_path = os.path.join(timestamp_dir, "graphs")
if not os.path.exists(graph_path):
    os.makedirs(graph_path)

#Graphing function
def graph(dist,p1,p2):
    # Calculate the time array
    time_array = [i / fps_real for i in range(len(dist))]
    plt.ylim(30, 130)
    # Plot the distance values for the first distance in the array
    plt.plot(time_array, dist)
    
    # Add title and labels
    plt.title(f'Euclidean Distance of Landmarks {p1} and {p2}')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (mm)')

    # Configure gridlines
    plt.grid(True, linewidth=0.5)
    plt.minorticks_on()
    plt.gca().set_axisbelow(True)
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))  # Set minor tick spacing to 0.1
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    # Adjust the aspect ratio to create smaller grid boxes
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3)  # Micro gridlines
    output_file = f'{graph_path}/front_euclidean_distance_{p1}_{p2}.jpg'
    
    # Save the plot as an image file
    plt.savefig(output_file,dpi=900)
    plt.clf()

# Initialize the variables
frame_count = 0
start_time = datetime.datetime.now()
graph_1 = []
graph_2 = []
first_face_position = None
elapsed_time_real=0
# Initialize point for displaying average of landmark points 22 and 23
avg_point = None
fps = 0
# Function for drawing lines
def draw_lines(frame):
    if show_lines:
        if show_lines_Sn_sPog:
            #Orange
            cv2.line(frame, point_34, point_9, (0, 125, 255), 1, cv2.LINE_AA)
        if show_lines_sN_Sn:
            if avg_point:
                #Blue
                cv2.line(frame, avg_point, point_34, (255, 0, 0), 1, cv2.LINE_AA)
        #DarkGreen
        cv2.line(frame, point_22, point_23, (0, 100, 0), 1, cv2.LINE_AA)
        if show_lines_Ar_Right:
            #Cyan
            cv2.line(frame, point_3, point_4, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, point_4, point_5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, point_5, point_6, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, point_6, point_7, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, point_7, point_8, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, point_8, point_9, (255, 255, 0), 1, cv2.LINE_AA)
        if show_lines_Ar_Left:
            #Red
            cv2.line(frame, point_9, point_10, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(frame, point_10, point_11, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(frame, point_11, point_12, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(frame, point_12, point_13, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(frame, point_13, point_14, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(frame, point_14, point_15, (0, 0, 255), 1, cv2.LINE_AA)
        if show_lines_iC_Left:
            #Purple
            cv2.line(frame, point_40, point_8, (255, 0, 255), 1, cv2.LINE_AA)
        if show_lines_iC_Right:
            #Light Green
            cv2.line(frame, point_43, point_10, (0, 255, 125), 1, cv2.LINE_AA)
        if show_lines_sZy:
            #Drawing Lines
            cv2.line(frame, point_1, point_17, (255, 255, 255), 2)

# Starting the Checkbox app
app = QApplication(sys.argv)
window = Checkbox()
window.show()
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
    box_width = int(width / 6)  # Adjust the fraction as needed
    box_height = int(height / 14)  # Adjust the fraction as needed
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
        if show_lines and show_lines_sZy:
            x1, y1 = landmarks.part(16).x, landmarks.part(16).y
            x2, y2 = landmarks.part(0).x, landmarks.part(0).y
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
        #Drawing dotted lines for point pairs
        point_1 = (landmarks.part(1-1).x, landmarks.part(1-1).y)
        point_2 = (landmarks.part(2-1).x, landmarks.part(2-1).y)
        point_3 = (landmarks.part(3-1).x, landmarks.part(3-1).y)
        point_4 = (landmarks.part(4-1).x, landmarks.part(4-1).y)
        point_5 = (landmarks.part(5-1).x, landmarks.part(5-1).y)
        point_6 = (landmarks.part(6-1).x, landmarks.part(6-1).y)
        point_7 = (landmarks.part(7-1).x, landmarks.part(7-1).y)
        point_8 = (landmarks.part(8-1).x, landmarks.part(8-1).y)
        point_9 = (landmarks.part(9-1).x, landmarks.part(9-1).y)
        point_10 = (landmarks.part(10-1).x, landmarks.part(10-1).y)
        point_11 = (landmarks.part(11-1).x, landmarks.part(11-1).y)
        point_12 = (landmarks.part(12-1).x, landmarks.part(12-1).y)
        point_13 = (landmarks.part(13-1).x, landmarks.part(13-1).y)
        point_14 = (landmarks.part(14-1).x, landmarks.part(14-1).y)
        point_15 = (landmarks.part(15-1).x, landmarks.part(15-1).y)
        point_17 = (landmarks.part(17-1).x, landmarks.part(17-1).y)
        point_22 = (landmarks.part(22-1).x, landmarks.part(22-1).y)
        point_23 = (landmarks.part(23-1).x, landmarks.part(23-1).y)
        point_34 = (landmarks.part(34-1).x, landmarks.part(34-1).y)
        point_40 = (landmarks.part(40-1).x, landmarks.part(40-1).y)
        point_43 = (landmarks.part(43-1).x, landmarks.part(43-1).y)
        
        # draw the lines on the frame
        draw_lines(frame)    
        
        # Loop through each landmark point
        for i in range(68):
            # Get the x,y coordinates of the landmark point
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            # Draw a circle around the landmark point
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # If landmark is 21 or 22, update avg_point
            if i == 21:
                x21, y21 = x, y
            elif i == 22:
                x22, y22 = x, y
                avg_point = ((x21 + x22) // 2, (y21 + y22) // 2)

            # Display average point on frame
            if avg_point:
                cv2.circle(frame, (avg_point[0],avg_point[1]), 3, (0, 255, 0), -1)

            # Write the landmark point coordinates to the landmark CSV file
            #with open(landmark_csv_path, mode='a') as landmark_file:
            #    landmark_writer = csv.writer(landmark_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #    landmark_writer.writerow([frame_count, i, x, y])

        # Calculate the eucledian distance between three specific landmark points
        d1 = math.sqrt((avg_point[0] - landmarks.part(34-1).x) ** 2 +
                       (avg_point[1] - landmarks.part(34-1).y) ** 2)
        d2 = math.sqrt((landmarks.part(34-1).x - landmarks.part(9-1).x) ** 2 +
                       (landmarks.part(34-1).y - landmarks.part(9-1).y) ** 2)
        d3 = math.sqrt((landmarks.part(40-1).x - landmarks.part(8-1).x) ** 2 +
                       (landmarks.part(40-1).y - landmarks.part(8-1).y) ** 2)
        d4 = math.sqrt((landmarks.part(43-1).x - landmarks.part(10-1).x) ** 2 +
                       (landmarks.part(43-1).y - landmarks.part(10-1).y) ** 2)
        d_3_4 = math.sqrt((landmarks.part(3-1).x - landmarks.part(4-1).x) ** 2 +
                       (landmarks.part(3-1).y - landmarks.part(4-1).y) ** 2)
        d_4_5 = math.sqrt((landmarks.part(4-1).x - landmarks.part(5-1).x) ** 2 +
                       (landmarks.part(4-1).y - landmarks.part(5-1).y) ** 2)
        d_5_6 = math.sqrt((landmarks.part(5-1).x - landmarks.part(6-1).x) ** 2 +
                       (landmarks.part(5-1).y - landmarks.part(6-1).y) ** 2)
        d_6_7 = math.sqrt((landmarks.part(6-1).x - landmarks.part(7-1).x) ** 2 +
                       (landmarks.part(6-1).y - landmarks.part(7-1).y) ** 2)
        d_7_8 = math.sqrt((landmarks.part(7-1).x - landmarks.part(8-1).x) ** 2 +
                       (landmarks.part(7-1).y - landmarks.part(8-1).y) ** 2)
        d_8_9 = math.sqrt((landmarks.part(8-1).x - landmarks.part(9-1).x) ** 2 +
                       (landmarks.part(8-1).y - landmarks.part(9-1).y) ** 2)
        d_9_10 = math.sqrt((landmarks.part(9-1).x - landmarks.part(10-1).x) ** 2 +
                       (landmarks.part(9-1).y - landmarks.part(10-1).y) ** 2)
        d_10_11 = math.sqrt((landmarks.part(10-1).x - landmarks.part(11-1).x) ** 2 +
                       (landmarks.part(10-1).y - landmarks.part(11-1).y) ** 2)
        d_11_12 = math.sqrt((landmarks.part(11-1).x - landmarks.part(12-1).x) ** 2 +
                       (landmarks.part(11-1).y - landmarks.part(12-1).y) ** 2)
        d_12_13 = math.sqrt((landmarks.part(12-1).x - landmarks.part(13-1).x) ** 2 +
                       (landmarks.part(12-1).y - landmarks.part(13-1).y) ** 2)
        d_13_14= math.sqrt((landmarks.part(13-1).x - landmarks.part(14-1).x) ** 2 +
                       (landmarks.part(13-1).y - landmarks.part(14-1).y) ** 2)
        d_14_15 = math.sqrt((landmarks.part(14-1).x - landmarks.part(15-1).x) ** 2 +
                       (landmarks.part(14-1).y - landmarks.part(15-1).y) ** 2)
        

        # Convert the distances from pixels to cm 
        face_width_px = math.sqrt((landmarks.part(17-1).x - landmarks.part(1-1).x) ** 2 +
                       (landmarks.part(17-1).y - landmarks.part(1-1).y) ** 2)
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
        
        if not isinstance(path, int) or recording_status:
            # Building array for the graph
            graph_1.append((d1_mm))
            graph_2.append((d2_mm))

            #Write the frame number and eucledian distances to the distance CSV file
            with open(distance_csv_path, mode='a') as distance_file:
                distance_writer = csv.writer(distance_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                distance_writer.writerow([frame_count,elapsed_time_real,face_width_mm, d1_mm, d2_mm, d3_mm, d4_mm, d_3_9_sum, d_9_15_sum])

        # Display the frame number and fps on the top left side of the frame
        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        fps = frame_count / elapsed_time
        
        #print(fps_real)
        # Calculate the elapsed time
        elapsed_time_real = frame_count/ fps_real
        
        

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

        
        # calculate the text size
        text_scale = 0.8  # Adjust the scale as needed
        text_thickness = 2
        text_font = cv2.FONT_HERSHEY_SIMPLEX

        text_width, text_height = cv2.getTextSize(text1, text_font, text_scale, text_thickness)[0]

        # calculate the text positions
        x_ec = frame.shape[1] - text_width - 10  # Adjust the offset as needed
        y_ec = int(frame.shape[0] * 0.05)  # Adjust the fraction as needed

        # draw the text on the frame
        cv2.putText(frame, text0_label, (x_ec, y_ec), text_font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text0, (x_ec, y_ec + int(text_height) + 10), text_font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text1, (x_ec, y_ec + int(text_height) * 2 + 20), text_font, text_scale, (255, 0, 0), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text2_label, (x_ec, y_ec + int(text_height) * 4 + 40), text_font, text_scale, (0, 125, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text2, (x_ec, y_ec + int(text_height) * 5 + 50), text_font, text_scale, (0, 125, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text3, (x_ec, y_ec + int(text_height) * 6 + 60), text_font, text_scale, (255, 0, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text4, (x_ec, y_ec + int(text_height) * 7 + 70), text_font, text_scale, (0, 255, 125), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text5_label, (x_ec, y_ec + int(text_height) * 8 + 90), text_font, text_scale, (255, 255, 0), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text5, (x_ec, y_ec + int(text_height) * 9 + 100), text_font, text_scale, (255, 255, 0), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text6_label, (x_ec, y_ec + int(text_height) * 10 + 120), text_font, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, text6, (x_ec, y_ec + int(text_height) * 11 + 130), text_font, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)

    else:
        # If no faces are detected, reset the first_face_position variable
        first_face_position = None

    text_scale = 0.8  # Adjust the scale as needed
    text_thickness = 2
    # Display fps, frame no and time
    cv2.putText(frame, f"Frame: {frame_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Time: {elapsed_time_real:.2f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
    if recording_status and frame_count%2==0:
        cv2.putText(frame, f"Recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
    

    # Display the frame
    cv2.imshow("Front Face Landmark Detection", frame)
    #cv2.createButton("Toggle Lines", toggle_lines, None, cv2.QT_CHECKBOX, False)
    
    if not isinstance(path, int) or recording_status:
        
        out.write(frame)
        # Save a snapshot of the GUI as an image
        cv2.imwrite(f"{images_dir}/{frame_count}.jpg", frame)
    
    
    
    # Wait for a key press to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27 or cv2.getWindowProperty("Front Face Landmark Detection", cv2.WND_PROP_VISIBLE) < 1:
        break
    

# Generating graphs    
graph(graph_1,'Soft tissue over nasion','Subnasale')
graph(graph_2,'Subnasale','Soft tissue over Poginion')
out.release()   
cap.release()
cv2.destroyAllWindows()
