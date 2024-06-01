import sys
import cv2
import torch
import os
import pyautogui
import numpy as np
import json
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer, Qt, QDateTime
from PyQt5.QtGui import QColor, QImage, QPixmap

# Function to perform inference
def run_yolo_inference(frame, model):
    results = model(frame)
    return results

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BRACU ALTER")
        self.setGeometry(100, 100, 1200, 800)

        # Set window colors using stylesheets
        self.setStyleSheet("background-color: #1C1C1C; color: white;")

        # Heading label
        heading_label = QLabel("BRACU ALTER", self)
        heading_label.setStyleSheet("color: #3498DB; font-size: 24pt; font-weight: bold;")
        heading_label.setAlignment(Qt.AlignCenter)
        heading_label.setGeometry(0, 10, self.width(), 40)

        # Main layout
        main_layout = QHBoxLayout()

        # Option section layout (left side)
        option_layout = QVBoxLayout()

        button_style = "background-color: black; color: white; font-size: 15pt; font-family: Arial; border: 2px solid blue;"

        # Button to select model
        self.select_model_button = QPushButton("Select Model")
        self.select_model_button.setStyleSheet(button_style)
        self.select_model_button.clicked.connect(self.select_model)
        option_layout.addWidget(self.select_model_button)

        # Button to start YOLOv5 detection
        self.detect_button = QPushButton("Start Detection")
        self.detect_button.setStyleSheet(button_style)
        self.detect_button.clicked.connect(self.start_detection)
        option_layout.addWidget(self.detect_button)

        # Button to stop YOLOv5 detection
        self.stop_detection_button = QPushButton("Close Detection")
        self.stop_detection_button.setStyleSheet(button_style)
        self.stop_detection_button.clicked.connect(self.close_detection)
        option_layout.addWidget(self.stop_detection_button)

        # Button to start screen recording
        self.record_button = QPushButton("Start Screen Recording")
        self.record_button.setStyleSheet(button_style)
        self.record_button.clicked.connect(self.start_screen_recording)
        option_layout.addWidget(self.record_button)

        # Button to save screen recording
        self.save_button = QPushButton("Save Recording")
        self.save_button.setStyleSheet(button_style)
        self.save_button.clicked.connect(self.save_screen_recording)
        option_layout.addWidget(self.save_button)

        # Button to take screenshot
        self.screenshot_button = QPushButton("Take Screenshot")
        self.screenshot_button.setStyleSheet(button_style)
        self.screenshot_button.clicked.connect(self.take_screenshot)
        option_layout.addWidget(self.screenshot_button)

        main_layout.addLayout(option_layout)

        # Camera feed layout (top middle)
        camera_layout = QVBoxLayout()

        # Camera feed label
        self.camera_feed_label = QLabel(self)
        self.camera_feed_label.setAlignment(Qt.AlignCenter)
        self.camera_feed_label.setFixedSize(800, 600)  # Adjust the size as needed
        camera_layout.addWidget(self.camera_feed_label)

        # Timer label
        self.timer_label = QLabel("00:00:00", self)
        self.timer_label.setStyleSheet("color: #3498DB; font-size: 20pt; font-weight: bold;")
        self.timer_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.timer_label)

        # Processing details layout (bottom middle)
        processing_details_layout = QVBoxLayout()

        # Label for processing details
        processing_details_label = QLabel("Processing Details:")
        processing_details_label.setStyleSheet("color: white; font-size: 15pt; font-family: roboto;")
        processing_details_layout.addWidget(processing_details_label)

        # Text edit to show processing details
        self.processing_details_output = QTextEdit()
        self.processing_details_output.setStyleSheet("background-color: #212F3D; color: white; font-size: 16pt; font-family: Arial;")
        processing_details_layout.addWidget(self.processing_details_output)

        camera_layout.addLayout(processing_details_layout)

        main_layout.addLayout(camera_layout)

        # Detected objects section layout (right side)
        detected_objects_layout = QVBoxLayout()

        # Table for detected objects
        self.detected_objects_table = QTableWidget()
        self.detected_objects_table.setRowCount(0)
        self.detected_objects_table.setColumnCount(3)
        self.detected_objects_table.setHorizontalHeaderLabels(['Class', 'Confidence', 'Time'])
        self.detected_objects_table.horizontalHeader().setStyleSheet("::section{Background-color: #3498DB; color: white;}")
        self.detected_objects_table.horizontalHeader().setStretchLastSection(True)
        detected_objects_layout.addWidget(self.detected_objects_table)

        main_layout.addLayout(detected_objects_layout)

        # Container widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Timer for updating camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_feed)
        self.cap = None

        # YOLO model
        self.model = None

        # Screen recording
        self.recording = False
        self.out = None

        # Timer for camera feed
        self.elapsed_time = 0
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_timer_label)

        # Detected objects storage
        self.detected_objects = set()

    def select_model(self):
        pt_file, _ = QFileDialog.getOpenFileName(self, "Select .pt file", "", "PyTorch Model Files (*.pt)")
        if pt_file:
            self.processing_details_output.append(f"Model selected: {pt_file}")
            self.load_model(pt_file)
        else:
            self.processing_details_output.append("No .pt file selected.")

    def start_detection(self):
        if self.model is None:
            self.processing_details_output.append("Error: No model loaded. Please load a .pt file first.")
            return

        self.processing_details_output.append("Starting YOLOv5 detection...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.processing_details_output.append("Error: Could not open camera.")
            return
        self.timer.start(30)
        self.elapsed_time = 0
        self.camera_timer.start(1000)  # Update every second

    def close_detection(self):
        self.processing_details_output.append("Closing YOLOv5 detection...")
        self.stop_camera_detection()
        self.save_detected_objects_to_json()

    def stop_camera_detection(self):
        self.processing_details_output.append("Stopping YOLOv5 camera detection...")
        self.timer.stop()
        self.camera_timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_camera_feed(self):
        if self.cap is None or not self.cap.isOpened():
            self.processing_details_output.append("Error: Camera is not initialized or could not be opened.")
            return

        ret, frame = self.cap.read()
        if ret:
            # Perform YOLOv5 inference
            results = run_yolo_inference(frame, self.model)
            # Draw results on frame
            frame = results.render()[0]
            # Convert frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to QImage
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # Update QLabel with QPixmap
            pixmap = QPixmap.fromImage(qt_image)
            # Adjust the size of the camera feed label
            self.camera_feed_label.setPixmap(pixmap.scaled(self.camera_feed_label.size(), Qt.KeepAspectRatio))
            # Log detected objects
            self.update_detected_objects_table(results)

            # Screen recording
            if self.recording and self.out is not None:
                # Convert frame back to BGR for recording
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.out.write(frame_bgr)
        else:
            self.processing_details_output.append("Error: Could not read frame from camera.")

    def update_timer_label(self):
        self.elapsed_time += 1
        elapsed_time_str = str(QDateTime.fromTime_t(self.elapsed_time).toUTC().toString("hh:mm:ss"))
        self.timer_label.setText(elapsed_time_str)

    def update_detected_objects_table(self, results):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            detected_object = {
                "class": str(results.names[int(cls)]),
                "confidence": f"{conf:.2f}",
                "time": current_time
            }
            if detected_object["class"] not in self.detected_objects:
                self.detected_objects.add(detected_object["class"])
                row_position = self.detected_objects_table.rowCount()
                self.detected_objects_table.insertRow(row_position)
                self.detected_objects_table.setItem(row_position, 0, QTableWidgetItem(detected_object["class"]))
                self.detected_objects_table.setItem(row_position, 1, QTableWidgetItem(detected_object["confidence"]))
                self.detected_objects_table.setItem(row_position, 2, QTableWidgetItem(detected_object["time"]))
    
    def update_detected_objects_graph(self):
        plt.clf()
        plt.bar(self.analysis_data.keys(), self.analysis_data.values(), color='blue')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Detected Object Counts by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('detected_objects_graph.png')


    def save_detected_objects_to_json(self):
        detected_objects_list = []
        for row in range(self.detected_objects_table.rowCount()):
            detected_object = {
                "class": self.detected_objects_table.item(row, 0).text(),
                "confidence": self.detected_objects_table.item(row, 1).text(),
                "time": self.detected_objects_table.item(row, 2).text()
            }
            detected_objects_list.append(detected_object)
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Detected Objects", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, "w") as file:
                json.dump(detected_objects_list, file, indent=4)
            self.processing_details_output.append(f"Detected objects saved to {file_path}")
    
    def analyze_detected_objects(self):
        plt.clf()
        plt.bar(self.analysis_data.keys(), self.analysis_data.values(), color='blue')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Detected Object Counts by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('detected_objects_graph.png')


    def save_analysis_as_pdf(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Analysis", "", "PDF Files (*.pdf)")
        if file_path:
            with PdfPages(file_path) as pdf:
                pdf.savefig()
                plt.close()
            self.processing_details_output.append(f"Analysis saved as PDF: {file_path}")


    def load_model(self, pt_file):
        import pathlib
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_file, force_reload=True)
            self.processing_details_output.append("Model loaded successfully.")
        except Exception as e:
            self.processing_details_output.append(f"Error loading model: {str(e)}")
        finally:
            pathlib.PosixPath = temp

    def start_screen_recording(self):
        self.processing_details_output.append("Starting screen recording...")
        self.recording = True
        # Prompt user to choose the save location
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Recording", "", "Video Files (*.avi)")
        if file_path:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(file_path, fourcc, 20.0, (640, 480))
            self.processing_details_output.append("Screen recording started.")
        else:
            self.processing_details_output.append("Screen recording cancelled.")

    def save_screen_recording(self):
        if self.out is not None:
            self.processing_details_output.append("Saving screen recording...")
            self.recording = False
            self.out.release()
            self.out = None
            self.processing_details_output.append("Screen recording saved successfully.")
        else:
            self.processing_details_output.append("No recording to save.")

    def take_screenshot(self):
        screenshot = pyautogui.screenshot()
        screenshot.save("screenshot.png")
        self.processing_details_output.append("Screenshot taken and saved as 'screenshot.png'.")

    def closeEvent(self, event):
        self.stop_camera_detection()
        self.save_screen_recording()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
