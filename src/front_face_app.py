#prototype  dlib v3
import csv
import datetime
import math
import os
import sys
import time
from multiprocessing import Process, Queue
from subprocess import Popen

import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import (QApplication, QCheckBox, QDesktopWidget,
                             QHBoxLayout, QPushButton, QVBoxLayout, QWidget)
from scipy import stats

# Initialize the webcam or video file path
#path = '..\sample\sample_front.mp4' #video file path or 0 for webcam
#path = 1
path = str(sys.argv[1])
if path=='None':
    path = 0

cap = cv2.VideoCapture(path)

if isinstance(path, int):
    # Set the desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



# Initialize the variables to toggle the lines
show_lines = True    
show_lines_sZy = False
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


# Get the path to the project's root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the outputs directory in the project's root directory
output_dir = os.path.join(project_root, '..' , 'outputs')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Timestamp Dir
if str(sys.argv[2]) is None:
    date_string = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M%p")
else:
    date_string = str(sys.argv[2])
    date_string = date_string[:-3]
timestamp_dir = os.path.join(output_dir, date_string)
if not os.path.exists(timestamp_dir):
    os.makedirs(timestamp_dir)

# Audio output directory
audio_output_dir = os.path.join(timestamp_dir, 'audio')
if not os.path.exists(audio_output_dir):
    os.makedirs(audio_output_dir)

# Set the audio output path
#audio_output_path = os.path.join(audio_output_dir, "raw.wav")
#video = VideoFileClip(path)
#audio = video.audio
#audio.write_audiofile(audio_output_path)

#sr, audio_data = wavfile.read(audio_output_path)
#denoised_audio = librosa.effects.decompose.noise(audio_data, frame_length=2048, hop_length=512)
#wavfile.write(audio_output_path, sr, denoised_audio)
    
# Get the path to the models directory
models_dir = os.path.join(project_root, '..', 'models')

# Load the shape predictor model from the models directory
shape_predictor_path = os.path.join(models_dir,'shape_predictor_68_face_landmarks.dat')

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

# Get the frame rate and resolution
fps_real = cap.get(cv2.CAP_PROP_FPS)
v_width = 640 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
v_height = 480 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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


def calculate_head_tilt(landmarks):
    nose_left = landmarks.part(27).x
    nose_right = landmarks.part(29).x

    if nose_left == nose_right:
        tilt_direction = "Straight"
        angle_degrees = 0
    else:
        horizontal_distance = nose_right - nose_left
        reference_distance = abs(landmarks.part(28).y - landmarks.part(27).y)

        angle_radians = math.atan(horizontal_distance / reference_distance)
        angle_degrees = math.degrees(angle_radians)

        if horizontal_distance > 0:
            tilt_direction = "Tilted Left"
        else:
            tilt_direction = "Tilted Right"
    
    return abs(angle_degrees), tilt_direction

# Initialize the variables
frame_count = 0
start_time = datetime.datetime.now()
graph_1 = []
graph_2 = []
first_face_position = None
elapsed_time_real=0
diameters =[]
diameter = None
reference_diameter_mm = float(sys.argv[3])  # The real-life diameter of the reference object in mm
conversion_rate = None
list_landmarks_9 = []
list_landmarks_34 = []
list_landmarks_31 = []
list_index = 0
theta= None
theta_2= None
horizontal_distance_moved = None
horizontal_distance_moved_2 = None
horz_dist_list = []
horz_dist_list_2 = []
highest_horz_dist = None
highest_horz_dist_2 = None
lowest_horz_dist = None
lowest_horz_dist_2 = None
# Initialize point for displaying average of landmark points 22 and 23
avg_point = None
fps = 0
# Function for drawing lines
def draw_lines(frame):
    if show_lines:
        if show_lines_Sn_sPog:
            #Orange
            cv2.line(frame, point_34, point_9, (0, 125, 255), 1, cv2.LINE_AA)
            cv2.line(frame, point_31d, point_31, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(frame, point_31d, point_9d, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(frame, point_9d, point_9, (0, 0, 255), 1, cv2.LINE_AA)
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
            #Brown
            cv2.line(frame, point_9, point_10, (165, 42, 42), 1, cv2.LINE_AA)
            cv2.line(frame, point_10, point_11, (165, 42, 42), 1, cv2.LINE_AA)
            cv2.line(frame, point_11, point_12, (165, 42, 42), 1, cv2.LINE_AA)
            cv2.line(frame, point_12, point_13, (165, 42, 42), 1, cv2.LINE_AA)
            cv2.line(frame, point_13, point_14, (165, 42, 42), 1, cv2.LINE_AA)
            cv2.line(frame, point_14, point_15, (165, 42, 42), 1, cv2.LINE_AA)
        if show_lines_iC_Left:
            #Purple
            cv2.line(frame, point_40, point_8, (255, 0, 255), 1, cv2.LINE_AA)
        if show_lines_iC_Right:
            #Lime Green
            cv2.line(frame, point_43, point_10, (50, 205, 50), 1, cv2.LINE_AA)
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
    box_width = int(width / 5)  # Adjust the fraction as needed
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
        
        # Get the forehead region based on landmarks
        forehead_top = landmarks.part(21).y - 70  # Adjust the value as needed
        forehead_bottom = landmarks.part(19).y  # Adjust the value as needed
        forehead_left = landmarks.part(19).x  # Adjust the value as needed
        forehead_right = landmarks.part(24).x  # Adjust the value as needed
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Extract the forehead region from the grayscale image
        #forehead_roi = blurred[forehead_top:forehead_bottom, forehead_left:forehead_right]
        # Create a blank mask with the same size as the image
        mask = np.zeros_like(gray)

        # Define the ROI using the facial landmarks
        roi_points = np.array([[(forehead_left, forehead_top),
                            (forehead_left, forehead_bottom),
                            (forehead_right, forehead_bottom),
                            (forehead_right, forehead_top)]], dtype=np.int32)

        # Fill the ROI region in the mask with white
        cv2.fillPoly(mask, roi_points, 255)

        # Apply the mask to the grayscale image
        masked_image = cv2.bitwise_and(gray, mask)
        circles = cv2.HoughCircles(masked_image, cv2.HOUGH_GRADIENT, dp=1, minDist=1000, param1=10, param2=20, minRadius=3, maxRadius=10)
        # Ensure that circles are detected
        if circles is not None:
            # Convert the circle parameters to integers
            circles = np.round(circles[0, :]).astype(int)
            
            # Get the first circle
            (x, y, r) = circles[0]
            
            diameter = r*2

             # Add the diameter to the list
            diameters.append(diameter)
                    # Calculate the mode diameter
            if diameters:
                mode_diameter = stats.mode(diameters, keepdims=True).mode[0]
                conversion_rate = reference_diameter_mm / mode_diameter
            # Convert circle coordinates to global coordinates
            #x += forehead_left
            #y += forehead_top
            # Calculate the conversion rate (pixels to mm) based on the diameter of the reference object
            
            # Draw the circle on the original image
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        # Define the forehead region polygon based on landmarks
        points = []
        points.append((landmarks.part(19).x, landmarks.part(19).y))  # Adjust the point indices as needed
        points.append((landmarks.part(24).x, landmarks.part(19).y))  # Adjust the point indices as needed
        points.append((landmarks.part(24).x, landmarks.part(21).y - 70))  # Adjust the point indices and the y-offset as needed
        points.append((landmarks.part(19).x, landmarks.part(21).y - 70))  # Adjust the point indices and the y-offset as needed
            # Convert the list of points to a NumPy array
        points = np.array(points)
        
        # Draw the forehead region polygon on the image
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=1)

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
        point_9d = (landmarks.part(9-1).x+10, landmarks.part(9-1).y)
        point_10 = (landmarks.part(10-1).x, landmarks.part(10-1).y)
        point_11 = (landmarks.part(11-1).x, landmarks.part(11-1).y)
        point_12 = (landmarks.part(12-1).x, landmarks.part(12-1).y)
        point_13 = (landmarks.part(13-1).x, landmarks.part(13-1).y)
        point_14 = (landmarks.part(14-1).x, landmarks.part(14-1).y)
        point_15 = (landmarks.part(15-1).x, landmarks.part(15-1).y)
        point_17 = (landmarks.part(17-1).x, landmarks.part(17-1).y)
        point_22 = (landmarks.part(22-1).x, landmarks.part(22-1).y)
        point_23 = (landmarks.part(23-1).x, landmarks.part(23-1).y)
        point_31 = (landmarks.part(31-1).x, landmarks.part(31-1).y)
        point_31d = (landmarks.part(31-1).x+10, landmarks.part(31-1).y)
        point_34 = (landmarks.part(34-1).x, landmarks.part(34-1).y)
        point_40 = (landmarks.part(40-1).x, landmarks.part(40-1).y)
        point_43 = (landmarks.part(43-1).x, landmarks.part(43-1).y)
        
        angle, tilt_direction = calculate_head_tilt(landmarks)
        
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
        d5 = math.sqrt((landmarks.part(31-1).x - landmarks.part(9-1).x) ** 2 +
                        (landmarks.part(31-1).y - landmarks.part(9-1).y) ** 2)
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
        
        #angle_1 = calculate_angle(point)
        
        # d1_mm = (d1 / face_width_px) * face_width_mm
        # d2_mm = (d2 / face_width_px) * face_width_mm
        # d3_mm = (d3 / face_width_px) * face_width_mm
        # d4_mm = (d4 / face_width_px) * face_width_mm

        # d_3_4_mm = (d_3_4 / face_width_px) * face_width_mm
        # d_4_5_mm = (d_4_5 / face_width_px) * face_width_mm
        # d_5_6_mm = (d_5_6 / face_width_px) * face_width_mm
        # d_6_7_mm = (d_6_7 / face_width_px) * face_width_mm
        # d_7_8_mm = (d_7_8 / face_width_px) * face_width_mm
        # d_8_9_mm = (d_8_9 / face_width_px) * face_width_mm
        # d_3_9_sum = d_3_4_mm + d_4_5_mm + d_5_6_mm + d_6_7_mm + d_7_8_mm + d_8_9_mm
        
        # d_9_10mm = (d_9_10 / face_width_px) * face_width_mm
        # d_10_11mm = (d_10_11 / face_width_px) * face_width_mm
        # d_11_12mm = (d_11_12 / face_width_px) * face_width_mm
        # d_12_13mm = (d_12_13 / face_width_px) * face_width_mm
        # d_13_14mm = (d_13_14 / face_width_px) * face_width_mm
        # d_14_15mm = (d_14_15 / face_width_px) * face_width_mm
        # d_9_15_sum = d_9_10mm + d_10_11mm + d_11_12mm + d_12_13mm + d_13_14mm + d_14_15mm
        
        if diameters or circles is not None:

            d1_mm = d1 * conversion_rate
            d2_mm = d2 * conversion_rate
            d3_mm = d3 * conversion_rate
            d4_mm = d4 * conversion_rate
            d5_mm = d5 * conversion_rate

            d_3_4_mm = d_3_4 * conversion_rate
            d_4_5_mm = d_4_5 * conversion_rate
            d_5_6_mm = d_5_6 * conversion_rate
            d_6_7_mm = d_6_7 * conversion_rate
            d_7_8_mm = d_7_8 * conversion_rate
            d_8_9_mm = d_8_9 * conversion_rate

            d_3_9_sum = d_3_4_mm + d_4_5_mm + d_5_6_mm + d_6_7_mm + d_7_8_mm + d_8_9_mm

            d_9_10_mm = d_9_10 * conversion_rate
            d_10_11_mm = d_10_11 * conversion_rate
            d_11_12_mm = d_11_12 * conversion_rate
            d_12_13_mm = d_12_13 * conversion_rate
            d_13_14_mm = d_13_14 * conversion_rate
            d_14_15_mm = d_14_15 * conversion_rate

            d_9_15_sum = d_9_10_mm + d_10_11_mm + d_11_12_mm + d_12_13_mm + d_13_14_mm + d_14_15_mm
            face_width_mm = face_width_px * conversion_rate
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
            
            import math

            if frame_count % 60 == 0:
                list_landmarks_34.append(point_34)
                list_landmarks_31.append(point_31)
                list_landmarks_9.append(point_9)
                list_index += 1
            # Assuming you have point_8 and point_34 as tuples containing (x, y) coordinates
            if list_landmarks_9 and list_landmarks_34 is not None:
                x9_initial, y9_initial = list_landmarks_9[list_index-1]
                x34_initial, y34_initial = list_landmarks_34[list_index-1]
                x31_initial, y31_initial = list_landmarks_31[list_index-1]
                

                # Calculate the initial displacement of Sn-sPog
                #initial_displacement = math.sqrt((x34_initial - x9_initial)**2 + (y34_initial - y9_initial)**2)
                #initial_displacement_2 = math.sqrt((x31_initial - x9_initial)**2 + (y31_initial - y9_initial)**2)
                # Assuming you have the new positions after jaw movement as point_8_new and point_34_new
                x9_final, y9_final = point_9
                x34_final, y34_final = point_34
                x31_final, y31_final = point_31

                initial_line_vector = (x34_final - x9_initial, y34_final - y9_initial) 
                final_line_vector = (x34_final - x9_final, y34_final - y9_final)

                # Calculate the dot product between the direction vectors
                dot_product = initial_line_vector[0] * final_line_vector[0] + initial_line_vector[1] * final_line_vector[1]
                # Calculate the magnitudes of the direction vectors
                initial_magnitude = math.sqrt(initial_line_vector[0] ** 2 + initial_line_vector[1] ** 2)
                final_magnitude = math.sqrt(final_line_vector[0] ** 2 + final_line_vector[1] ** 2)

                # Calculate the angle between the two lines in radians
                angle_radians = np.arccos(np.clip(dot_product / (initial_magnitude * final_magnitude), -1.0, 1.0))

                # Convert the angle to degrees
                theta = math.degrees(angle_radians)

                # Calculate the vertical displacement of the chin landmark after considering the angle
                horizontal_displacement = (x9_final - x9_initial) * math.sin(angle_radians)

                # Calculate the vertical displacement of the chin landmark after considering the angle
                vertical_displacement = (y9_final - y9_initial) * math.sin(angle_radians)

                # Calculate the distance between initial and final chin landmarks after considering the angle
                distance_after_angle = math.sqrt(horizontal_displacement ** 2 + vertical_displacement ** 2)
                # Calculate the final displacement of Sn-sPog after jaw movement
                #final_displacement = x9_final - x9_initial
                if horizontal_displacement < 0:
                    distance_after_angle = distance_after_angle*-1
                #dx = abs(abs(x9_final) - abs(x9_initial))
                #dy = abs(abs(y9_final) - abs(y9_initial))
                # Calculate the horizontal distance moved using the final displacement
                #horizontal_distance_moved = final_displacement
                horizontal_distance_moved = distance_after_angle*conversion_rate
                #final_displacement = final_displacement*conversion_rate


                #horizontal_distance_moved = horizontal_distance_moved*conversion_rate
                horz_dist_list.append(horizontal_distance_moved)
                highest_horz_dist = max(horz_dist_list)
                lowest_horz_dist = abs(min(horz_dist_list))

                # Calculate the angle theta in radians (as the angle with respect to the vertical axis)
                #theta = math.atan2(dy,dx)
                #theta = math.degrees(theta)

                # 31 Line 
                #theta = calculate_angle(list_landmarks_9[list_index-1],point_9,point_34)
                initial_line_vector_2 = (x31_final - x9_initial, y31_final - y9_initial) 
                final_line_vector_2 = (x31_final - x9_final, y31_final - y9_final)

                # Calculate the dot product between the direction vectors
                dot_product_2 = initial_line_vector_2[0] * final_line_vector_2[0] + initial_line_vector_2[1] * final_line_vector_2[1]
                # Calculate the magnitudes of the direction vectors
                initial_magnitude_2 = math.sqrt(initial_line_vector_2[0] ** 2 + initial_line_vector_2[1] ** 2)
                final_magnitude_2 = math.sqrt(final_line_vector_2[0] ** 2 + final_line_vector_2[1] ** 2)

                # Calculate the angle between the two lines in radians
                angle_radians_2 = np.arccos(np.clip(dot_product_2 / (initial_magnitude_2 * final_magnitude_2), -1.0, 1.0))

                # Convert the angle to degrees
                theta_2 = math.degrees(angle_radians_2)

                # Calculate the vertical displacement of the chin landmark after considering the angle
                horizontal_displacement_2 = (x9_final - x9_initial) * math.sin(angle_radians_2)

                # Calculate the vertical displacement of the chin landmark after considering the angle
                vertical_displacement_2 = (y9_final - y9_initial) * math.sin(angle_radians_2)

                # Calculate the distance between initial and final chin landmarks after considering the angle
                distance_after_angle_2 = math.sqrt(horizontal_displacement_2 ** 2 + vertical_displacement_2 ** 2)
                # Calculate the final displacement of Sn-sPog after jaw movement
                #final_displacement = x9_final - x9_initial
                if horizontal_displacement_2 < 0:
                    distance_after_angle_2 = distance_after_angle_2*-1
                #dx = abs(abs(x9_final) - abs(x9_initial))
                #dy = abs(abs(y9_final) - abs(y9_initial))
                # Calculate the horizontal distance moved using the final displacement
                #horizontal_distance_moved = final_displacement
                horizontal_distance_moved_2 = distance_after_angle_2*conversion_rate
                #final_displacement = final_displacement*conversion_rate


                #horizontal_distance_moved = horizontal_distance_moved*conversion_rate
                horz_dist_list_2.append(horizontal_distance_moved_2)
                highest_horz_dist_2 = max(horz_dist_list_2)
                lowest_horz_dist_2 = abs(min(horz_dist_list_2))

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

            text7= f"Sideways Angle: {theta}"
            text7d= f"Sideways Angle: {theta_2}"

            if horizontal_distance_moved is not None:
                text8= f"Sideways Dist(X): {horizontal_distance_moved:.2f}mm"
                
                text9= f"Left Dist(X):{highest_horz_dist:.2f}"

                text10= f"Right Dist(X):{lowest_horz_dist:.2f}"
                
                text8d= f"Sideways Dist(X): {horizontal_distance_moved_2:.2f}mm"
                
                text9d= f"Left Dist(X):{highest_horz_dist_2:.2f}"

                text10d= f"Right Dist(X):{lowest_horz_dist_2:.2f}"
            else:
                text8= f"Sideways Dist(X): N/A"
                text8d= f"Sideways Dist(X): N/A"
                
                text9= f"Left Dist(X): N/A"
                text9d= f"Left Dist(X): N/A"

                text10= f"Right Dist(X): N/A"
                text10d= f"Right Dist(X): N/A"
            # calculate the text size
            text_scale = 0.6  # Adjust the scale as needed
            text_thickness = 2
            text_font = cv2.FONT_HERSHEY_SIMPLEX

            text_width, text_height = cv2.getTextSize(text1, text_font, text_scale, text_thickness)[0]

            # calculate the text positions
            x_ec = frame.shape[1] - text_width -100  # Adjust the offset as needed
            y_ec = int(frame.shape[0] * 0.05)  # Adjust the fraction as needed

            # draw the text on the frame
            cv2.putText(frame, f"Ref_Diameter: {reference_diameter_mm}mm", (x_ec, y_ec), text_font, text_scale, (255,255,255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, f"Diameter: {mode_diameter}px", (x_ec, y_ec + int(text_height) + 10), text_font, text_scale, (255,255,255), text_thickness, cv2.LINE_AA)
            
            cv2.putText(frame, f"Tilt: {tilt_direction} ({angle:.2f} degrees)", (x_ec, y_ec + int(text_height) * 2 + 20), text_font, text_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text1, (x_ec, y_ec + int(text_height) * 3 + 30), text_font, text_scale, (255, 0, 0), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text2_label, (x_ec, y_ec + int(text_height) * 4 + 40), text_font, text_scale, (0, 125, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text2, (x_ec, y_ec + int(text_height) * 5 + 50), text_font, text_scale, (0, 125, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text3, (x_ec, y_ec + int(text_height) * 6 + 60), text_font, text_scale, (255, 0, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text4, (x_ec, y_ec + int(text_height) * 7 + 70), text_font, text_scale, (0, 255, 125), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text5_label, (x_ec, y_ec + int(text_height) * 8 + 90), text_font, text_scale, (255, 255, 0), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text5, (x_ec, y_ec + int(text_height) * 9 + 100), text_font, text_scale, (255, 255, 0), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text6_label, (x_ec, y_ec + int(text_height) * 10 + 120), text_font, text_scale, (165, 42, 42), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text6, (x_ec, y_ec + int(text_height) * 11 + 130), text_font, text_scale, (165, 42, 42), text_thickness, cv2.LINE_AA)
            
            cv2.putText(frame, text7, (x_ec, y_ec + int(text_height) * 13 + 150), text_font, text_scale, (0, 125, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text8, (x_ec, y_ec + int(text_height) * 14 + 160), text_font, text_scale, (0, 125, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text9, (x_ec, y_ec + int(text_height) * 15 + 170), text_font, text_scale, (0, 125, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text10, (x_ec, y_ec + int(text_height) * 16 + 180), text_font, text_scale, (0, 125, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(frame, text7d, (x_ec, y_ec + int(text_height) * 17 + 200), text_font, text_scale, (0,0, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text8d, (x_ec, y_ec + int(text_height) * 18 + 210), text_font, text_scale, (0,0, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text9d, (x_ec, y_ec + int(text_height) * 19 + 220), text_font, text_scale, (0,0, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, text10d, (x_ec, y_ec + int(text_height) * 20 + 230), text_font, text_scale, (0,0, 255), text_thickness, cv2.LINE_AA)
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
    
    #if not isinstance(path, int) or recording_status:
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
#Popen(['python', 'phonetics.py', str(path), str(date_string), str(distance_csv_path)])
#Popen(['python', 'process.py',str(path), str(date_string), str(distance_csv_path)])
cv2.destroyAllWindows()