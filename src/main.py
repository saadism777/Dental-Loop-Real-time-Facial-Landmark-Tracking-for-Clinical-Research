import os
import signal
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication,QLabel,QDesktopWidget,QHBoxLayout, QCheckBox, QVBoxLayout
import socket
import time
import sys
from subprocess import Popen
from PyQt5.QtCore import QTimer, Qt

#Initializign variables
front_proc = None
side_proc = None
front_face_width = None
side_face_width = None

def start_or_stop_processes(start):
    global front_proc, side_proc
    if start:
        # Start the two subprocesses and return the objects
        front_proc = Popen(['python', 'front_face_app.py', str(front_face_width)])
        side_proc = Popen(['python', 'side_face_app.py', str(side_face_width)])
        return front_proc, side_proc
    else:
        # Stop the two subprocesses using their objects
        # front_proc.send_signal(signal.SIGTERM)
        # side_proc.send_signal(signal.SIGTERM)
        print("Already running")

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.resize(300, 300)
        self.center()
        self.setWindowTitle("Dental Loop")
        self.label = QtWidgets.QLabel("Enter the front face width value:")
        self.entry = QtWidgets.QLineEdit()
        self.label2 = QtWidgets.QLabel("Enter the side face width value:")
        self.entry2 = QtWidgets.QLineEdit()
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label2.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setFont(font)
        self.label2.setFont(font)
        self.button = QtWidgets.QPushButton("OK", clicked=self.get_entry_value)
        self.button2 = QtWidgets.QPushButton("Exit", clicked=self.close_app)
        # Create the label and checkbox widgets
        #self.label3 = QLabel('Front Face Lines:', self)
        #self.checkbox3_1 = QCheckBox(self)
        #self.label3_1 = QLabel(': sZy left - sZy right', self)

        #self.checkbox3_2 = QCheckBox(self)
        #self.label3_2 = QLabel(':sN - Sn', self)


        #self.label4 = QLabel('Side Face Lines:', self)
        #self.label4_1 = QLabel(': sZy left - sZy right', self)
        #self.checkbox4 = QCheckBox(self)
        
        # Add a label to display the d1_mm variable
        #self.d1_mm_label = QtWidgets.QLabel("d1_mm: ")

        layout = QVBoxLayout(self)
        
        layout.addWidget(self.label)
        layout.addWidget(self.entry)
        layout.addWidget(self.label2)
        layout.addWidget(self.entry2)
        layout.addWidget(self.button)
        layout.addWidget(self.button2)
        #layout.addWidget(self.label3)
        
        #layoutH = QHBoxLayout(self)
        #layout.addLayout(layoutH)
        #layoutH.addWidget(self.checkbox3_1)
        #layoutH.addWidget(self.label3_1)

        #layoutH3 = QHBoxLayout(self)
        #layout.addLayout(layoutH3)
        #layoutH3.addWidget(self.checkbox3_2)
        #layoutH3.addWidget(self.label3_2)
        
        #layout.addWidget(self.label4)

        #layoutH2 = QHBoxLayout(self)
        #layout.addLayout(layoutH2)
        
        #layoutH2.addWidget(self.checkbox4)
        #layoutH2.addWidget(self.label4_1)
        
        #layout.addWidget(self.d1_mm_label)

        # Set the initial state of the checkboxes
        #self.checkbox3_1.setChecked(True)
        #self.checkbox3_2.setChecked(True)
        #self.checkbox4.setChecked(True)

        # Connect checkbox signals to respective slots
        #self.checkbox3_1.stateChanged.connect(self.handleCheckbox3_1)
        #self.checkbox3_2.stateChanged.connect(self.handleCheckbox3_2)
        #self.checkbox4.stateChanged.connect(self.handleCheckbox4)
        #self.queue = Queue()
        #self.subprocess = None
    
   
    #def handleCheckbox3_1(self, state):
    #        if self.subprocess is not None:
    #            if state == 2:  # Checked state
    #                self.queue.put('show_lines')
    #            else:
    #                self.queue.put('hide_lines')
    def center(self):
        # Get the geometry of the screen
        screen = QDesktopWidget().screenGeometry()

        # Calculate the center position of the screen
        center_x = screen.width() // 2
        center_y = screen.height() // 2

        # Move the window to the center of the screen
        self.move(center_x - self.width() // 2, center_y - self.height() // 2)
    def get_entry_value(self):
        global front_face_width, side_face_width
        front_face_width = float(self.entry.text())
        side_face_width = float(self.entry2.text())
        # Use the face_width value in your application
        print("The front face width is:", front_face_width)
        print("The side face width is:", side_face_width)
        # Launch both apps
        # Example usage:
        start_or_stop_processes(True)
        #self.start_subprocess()
    
    def close_app(self):
        # Stop the two subprocesses
        #if front_proc and side_proc is not None:
        #    start_or_stop_processes(False)
        timer = QTimer(self)
        timer.timeout.connect(QApplication.quit)
        timer.start(500)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
