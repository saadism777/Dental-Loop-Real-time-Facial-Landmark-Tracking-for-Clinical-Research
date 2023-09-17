import os
import signal
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication,QLabel,QDesktopWidget,QHBoxLayout, QCheckBox, QVBoxLayout, QFileDialog
import socket
import time
import sys
from subprocess import Popen
from PyQt5.QtCore import QTimer, Qt

#Initializign variables
front_proc = None
side_proc = None
marker_diameter = None
front_path = None
side_path = None


class MainWindow(QtWidgets.QWidget):
    def start_or_stop_processes(self):
        global front_proc, side_proc
        global marker_diameter
        marker_diameter = float(self.entry.text())

        # Use the face_width value in your application
        print(f"The marker diameter is: {marker_diameter}mm")
        # Start the two subprocesses and return the objects
        if side_path is not None:
            side_proc = Popen(['python', 'side_face_app.py', str(side_path),str(default_output_filename2), str(marker_diameter)])
        if front_path is not None:
            front_proc = Popen(['python', 'front_face_app.py', str(front_path), str(default_output_filename), str(marker_diameter)])

        
        
         
            
    def __init__(self):
        super().__init__()
        self.resize(300, 300)
        self.center()
        self.setWindowTitle("Dental Loop FLT")
        self.label = QtWidgets.QLabel("Enter tracking marker diameter value(mm):")
        self.entry = QtWidgets.QLineEdit()
        self.entry.setText("15")
        
        self.labelB = QLabel("Select Front Face Input File: ")
        self.buttonB = QtWidgets.QPushButton('Browse File')
        self.buttonB.clicked.connect(self.showFileDialog)
        
        #self.label2 = QtWidgets.QLabel("Enter the side face width value:")
        #self.entry2 = QtWidgets.QLineEdit()

        self.labelC = QLabel("Select Side Face Input File: ")
        self.buttonC = QtWidgets.QPushButton('Browse File')
        self.buttonC.clicked.connect(self.showFileDialog2)

        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        #self.label2.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setFont(font)
        #self.label2.setFont(font)
        self.button = QtWidgets.QPushButton("OK", clicked=self.start_or_stop_processes)
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
        layout.addWidget(self.labelB)
        layout.addWidget(self.buttonB)
        #layout.addWidget(self.label2)
        #layout.addWidget(self.entry2)
        layout.addWidget(self.labelC)
        layout.addWidget(self.buttonC)
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
    def showFileDialog(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.fileSelected.connect(self.fileSelectedAction)
        file_dialog.exec_()
    
    def fileSelectedAction(self, filepath):
        # Do something with the filepath
        global front_path, default_output_filename
        front_path=filepath
        self.labelB.setText(f"Selected file: {front_path}")
        # Extract the path from the last backslash up to the file extension
        filename = os.path.basename(filepath)
        index_of_last_backslash = filename.rfind("\\")
        index_of_extension = filename.rfind(".")
        if index_of_last_backslash != -1 and index_of_extension != -1:
            default_output_filename = filename[index_of_last_backslash + 1:index_of_extension]
        else:
            # If there's no backslash or no file extension, set the entire filename as default
                default_output_filename = filename

    def showFileDialog2(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.fileSelected.connect(self.fileSelectedAction2)
        file_dialog.exec_()
    def fileSelectedAction2(self, filepath):
            # Do something with the filepath
            global side_path, default_output_filename2
            side_path=filepath
            self.labelC.setText(f"Selected file: {side_path}")
            # Extract the path from the last backslash up to the file extension
            filename = os.path.basename(filepath)
            index_of_last_backslash = filename.rfind("\\")
            index_of_extension = filename.rfind(".")
            if index_of_last_backslash != -1 and index_of_extension != -1:
                default_output_filename2 = filename[index_of_last_backslash + 1:index_of_extension]
            else:
                # If there's no backslash or no file extension, set the entire filename as default
                default_output_filename2 = filename
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
        side_face_width = float(self.entry2.text()) if self.entry2.text() else None
        # Use the face_width value in your application
        print("The front face width is:", front_face_width)
        print("The side face width is:", side_face_width)
        # Launch both apps
        # Example usage:
        self.start_or_stop_processes()
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
