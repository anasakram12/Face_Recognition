# client.py
import sys
import os
import cv2
import base64
import requests
import configparser
import logging
from datetime import datetime
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, 
                           QTableWidget, QTableWidgetItem, QMenuBar, 
                           QAction, QInputDialog, QHeaderView, QWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer, QDateTime, QMimeData, QUrl
from PyQt5.QtGui import QPixmap, QDrag
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Server configuration
SERVER_URL = "http://192.168.15.65:3005"  # Replace with your server's IP address

# Setup logging
LOG_FILE = 'face_recognition_client_131.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DraggableLabel(QLabel):
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_path = file_path

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setUrls([QUrl.fromLocalFile(self.file_path)])
            drag.setMimeData(mime_data)
            drag.exec_(Qt.CopyAction | Qt.MoveAction)

class UnrecognizedLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QLabel {
                background-color: #3d3d3d;
                border: 2px solid #555;
                border-radius: 5px;
                padding: 5px;
                margin: 5px;
                color: #ff6b6b;
                font-weight: bold;
                font-size: 12px;
                min-width: 80px;
                max-width: 100px;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Unrecognized")
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

class PathLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 12px;
                padding: 5px;
                background-color: #2d2d2d;
                border-radius: 3px;
            }
        """)

class StatusLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-size: 14px;
                padding: 8px;
                background-color: #333333;
                border-radius: 5px;
                margin: 5px;
            }
        """)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setText("Status: Ready")
        self.setWordWrap(True)
        self.setMinimumHeight(40)

class FaceRecognitionClient(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.config_file = 'client_config.ini'
        self.config = self.load_config()
        self.watch_folder = self.config.get('Settings', 'watch_folder', fallback=None)
        self.path_label = None
        self.status_label = None
        self.init_ui()
        self.observer = Observer()
        self.recent_matches = []
        
        if self.watch_folder:
            self.start_watching_folder(silent=True)

    def load_config(self):
        config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            config['Settings'] = {'watch_folder': ''}
            with open(self.config_file, 'w') as f:
                config.write(f)
        else:
            config.read(self.config_file)
        return config

    def save_config(self):
        with open(self.config_file, 'w') as f:
            self.config.write(f)

    def init_ui(self):
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 1600, 800)
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: white;
                font-family: Arial, sans-serif;
            }
            QMenuBar {
                background-color: #2b2b2b;
                color: white;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QTableWidget {
                background-color: #252526;
                gridline-color: #444;
                color: white;
                font-size: 14px;
            }
            QTableWidget QHeaderView::section {
                background-color: #3d3d3d;
                color: white;
                font-weight: bold;
            }
            QPushButton {
                background-color: #3a6ea5;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #5591d2;
            }
            QLabel#timeLabel {
                font-size: 24px;
                font-weight: bold;
                color: #03DAC5;
            }
        """)

        main_layout = QVBoxLayout()

        # Menu Bar
        self.menu_bar = QMenuBar(self)
        file_menu = self.menu_bar.addMenu("File")
        
        # Add Select Folder Action
        select_folder_action = QAction("Select Folder", self)
        select_folder_action.triggered.connect(self.select_folder)
        file_menu.addAction(select_folder_action)

        # Add Reload Pickle Action
        reload_pickle_action = QAction("Reload Database", self)
        reload_pickle_action.triggered.connect(self.reload_embeddings)
        file_menu.addAction(reload_pickle_action)

        # Add Set Threshold Action
        set_threshold_action = QAction("Set Threshold", self)
        set_threshold_action.triggered.connect(self.set_threshold)
        file_menu.addAction(set_threshold_action)

        # Create horizontal layout for logo and title
        header_layout = QHBoxLayout()
        
        # Logo Label
        logo_label = QLabel()
        logo_pixmap = QPixmap('logo.png')
        if not logo_pixmap.isNull():
            scaled_logo = logo_pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
        logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Title Label
        title_label = QLabel("SmartFace Sentinel")
        title_label.setStyleSheet("font-size: 34px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Add logo and title to horizontal layout
        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.setAlignment(Qt.AlignCenter)
        header_layout.setSpacing(10)

        self.path_label = PathLabel("No folder selected")
        if self.watch_folder:
            self.path_label.setText(f"Watching: {self.watch_folder}")
        self.path_label.setAlignment(Qt.AlignLeft)

        # Status label for alerts
        self.status_label = StatusLabel()

        # Table for displaying last 10 inputs and outputs
        self.result_table = QTableWidget(5, 6)
        self.result_table.setHorizontalHeaderLabels([
            "Input Image ", "Recognized Image", "Details ",
            "Input Image ", "Recognized Image", "Details "
        ])
        self.result_table.horizontalHeader().setStretchLastSection(False)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setFixedHeight(600)
        
        for row in range(5):
            self.result_table.setRowHeight(row, 120)

        # Time Label
        self.time_label = QLabel()
        self.time_label.setObjectName("timeLabel")
        self.time_label.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        self.time_label.setStyleSheet(
            "position: absolute; margin-right: 20px; margin-bottom: 20px;"
        )

        # Update the time every second
        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)
        self.update_time()

        # Add all widgets to main layout
        main_layout.setMenuBar(self.menu_bar)
        main_layout.addLayout(header_layout)
        main_layout.addWidget(self.path_label)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.result_table)
        main_layout.addWidget(self.time_label)
        
        self.setLayout(main_layout)

    def update_time(self):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.time_label.setText(current_time)

    def show_status(self, message, error=False):
        # Show message in status label with proper styling
        if error:
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #ff6b6b;
                    font-size: 14px;
                    padding: 8px;
                    background-color: #333333;
                    border-radius: 5px;
                    margin: 5px;
                }
            """)
        else:
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-size: 14px;
                    padding: 8px;
                    background-color: #333333;
                    border-radius: 5px;
                    margin: 5px;
                }
            """)
        
        self.status_label.setText(f"Status: {message}")
        
        # Auto-clear status after 5 seconds
        QTimer.singleShot(5000, lambda: self.status_label.setText("Status: Ready"))

    def set_threshold(self):
        new_threshold, ok = QInputDialog.getDouble(
            self, "Set Threshold", "Enter new threshold (0.0 - 1.0):",
            0.5, 0.0, 1.0, 2
        )
        if ok:
            try:
                response = requests.post(
                    f"{SERVER_URL}/set_threshold",
                    json={"threshold": new_threshold}
                )
                if response.status_code == 200:
                    msg = f"Threshold has been set to {new_threshold:.2f}"
                    self.show_status(msg)
                    logging.info(msg)
            except requests.RequestException as e:
                error_msg = f"Failed to set threshold: {str(e)}"
                self.show_status(error_msg, error=True)
                logging.error(error_msg)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Watch")
        if folder:
            self.watch_folder = folder
            self.config.set('Settings', 'watch_folder', self.watch_folder)
            self.save_config()
            self.start_watching_folder()
            self.path_label.setText(f"Watching: {self.watch_folder}")

    def start_watching_folder(self, silent=False):
        if self.watch_folder:
            self.observer.stop()
            self.observer = Observer()
            event_handler = FolderWatchHandler(self)
            self.observer.schedule(event_handler, self.watch_folder, recursive=False)
            self.observer.start()
            
            msg = f"Now watching folder: {self.watch_folder}"
            self.show_status(msg)
            logging.info(msg)

    def reload_embeddings(self):
        try:
            response = requests.post(f"{SERVER_URL}/reload_embeddings")
            if response.status_code == 200:
                msg = "Face database has been reloaded successfully"
                self.show_status(msg)
                logging.info(msg)
        except requests.RequestException as e:
            error_msg = f"Failed to reload database: {str(e)}"
            self.show_status(error_msg, error=True)
            logging.error(error_msg)

    def process_new_image(self, file_path):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Read and encode image
            with open(file_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Check for camera number
            camera_number = None
            if 'camera01' in file_path:
                camera_number = 1
                color = QtGui.QColor('blue')
            elif 'camera02' in file_path:
                camera_number = 2
                color = QtGui.QColor('purple')
            
            # Send to server with camera ID
            payload = {
                "image": img_data,
            }
            
            # Add camera_id to payload if detected
            if camera_number:
                payload["camera_id"] = camera_number
            
            # Send request to server
            response = requests.post(
                f"{SERVER_URL}/recognize",
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Server error: {response.text}")
            
            result = response.json()
            
            # Create input image pixmap
            input_pixmap = QPixmap(file_path)
            input_pixmap = input_pixmap.scaled(
                100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # If camera number was detected, draw boxes
            if camera_number:
                # Draw a box around the image for the identified camera
                painter = QtGui.QPainter(input_pixmap)
                painter.setPen(QtGui.QPen(color, 3))  # 3px width for the box
                painter.drawRect(0, 0, input_pixmap.width()-1, input_pixmap.height()-1)  # Draw the rectangle
                painter.end()
                
                # Add the camera number as a label
                label = QLabel(f"Camera {camera_number}")
                label.setStyleSheet(f"background-color: {color.name()}; color: white; font-weight: bold;")
                label.setAlignment(Qt.AlignCenter)
                label.setFixedWidth(40)  # Set the fixed width of the label

                # Add the label and box to the image
                input_pixmap = self.add_label_to_pixmap(input_pixmap, label)

            if result['status'] == 'matched':
                # Decode matched image
                matched_image_data = base64.b64decode(result['matched_image'])
                matched_pixmap = QPixmap()
                matched_pixmap.loadFromData(matched_image_data)
                matched_pixmap = matched_pixmap.scaled(
                    100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                
                msg = f"Match found for image {os.path.basename(file_path)}: {result['matched_filename']} ({result['confidence']:.2%})"
                self.show_status(msg)
                logging.info(msg)
                
                self.update_recent_matches(
                    input_pixmap,
                    matched_pixmap,
                    timestamp,
                    result['matched_filename'],
                    result['confidence'],
                    file_path
                )
            else:
                msg = f"No match found for image {os.path.basename(file_path)}"
                self.show_status(msg)
                logging.info(msg)
                
                self.update_recent_matches(
                    input_pixmap,
                    "unrecognized",
                    timestamp,
                    None,
                    None,
                    file_path
                )
                
        except Exception as e:
            error_msg = f"Error processing image {file_path}: {str(e)}"
            self.show_status(error_msg, error=True)
            logging.error(error_msg)

    def add_label_to_pixmap(self, pixmap, label):
        # Create a new image with the label overlaid on the pixmap
        image = pixmap.toImage()
        painter = QtGui.QPainter(image)
        painter.setPen(QtGui.QPen(Qt.white, 1))
        painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        
        # Add label at the top-left corner
        painter.drawText(5, 15, label.text())
        painter.end()
        
        # Convert image back to pixmap
        return QPixmap.fromImage(image)
    
    def update_recent_matches(self, input_image_pixmap, output_image_pixmap,
                            timestamp, output_filename, confidence, input_file_path):
        details = f"Timestamp: {timestamp}"
        if output_filename:
            details += f"\nFile: {output_filename}"
        if confidence is not None:
            details += f"\nConfidence: {confidence:.2%}"
        else:
            details += "\nStatus: Unrecognized"

        self.recent_matches.insert(0, (
            input_image_pixmap, output_image_pixmap,
            details, input_file_path
        ))
        if len(self.recent_matches) > 10:
            self.recent_matches.pop()
        self.refresh_table()

    def refresh_table(self):
        try:
            # Clear all rows
            for i in range(5):
                for j in range(6):
                    self.result_table.setCellWidget(i, j, None)
                    self.result_table.setItem(i, j, QTableWidgetItem(""))

            for row in range(5):
                idx1 = row * 2
                idx2 = idx1 + 1
                
                for current_idx, column_start in [(idx1, 0), (idx2, 3)]:
                    if current_idx < len(self.recent_matches):
                        input_pixmap, output_pixmap, details, file_path = self.recent_matches[current_idx]
                        
                        # Input image
                        input_label = DraggableLabel(file_path)
                        input_label.setPixmap(input_pixmap)
                        input_label.setAlignment(Qt.AlignCenter)
                        self.result_table.setCellWidget(row, column_start, input_label)

                        # Output image/unrecognized label
                        if output_pixmap == "unrecognized":
                            unrecognized_label = UnrecognizedLabel()
                            cell_widget = QWidget()
                            layout = QHBoxLayout(cell_widget)
                            layout.addWidget(unrecognized_label)
                            layout.setAlignment(Qt.AlignCenter)
                            layout.setContentsMargins(0, 0, 0, 0)
                            self.result_table.setCellWidget(row, column_start + 1, cell_widget)
                        else:
                            output_label = QLabel()
                            if output_pixmap:
                                output_label.setPixmap(output_pixmap)
                                output_label.setAlignment(Qt.AlignCenter)
                            self.result_table.setCellWidget(row, column_start + 1, output_label)

                        # Details
                        details_item = QTableWidgetItem(details)
                        details_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                        self.result_table.setItem(row, column_start + 2, details_item)

        except Exception as e:
            error_msg = f"Error refreshing table: {str(e)}"
            self.show_status(error_msg, error=True)
            logging.error(error_msg)

    def closeEvent(self, event):
        """Handle application shutdown"""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
        logging.info("Application closed")
        event.accept()


class FolderWatchHandler(FileSystemEventHandler, QObject):
    image_created_signal = pyqtSignal(str)

    def __init__(self, app):
        FileSystemEventHandler.__init__(self)
        QObject.__init__(self)
        self.app = app
        self.image_created_signal.connect(self.app.process_new_image)

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.image_created_signal.emit(event.src_path)


def main():
    try:
        # Test server connection
        requests.get(f"{SERVER_URL}/")
    except requests.RequestException as e:
        # Write to log instead of showing popup
        logging.error(f"Server Connection Error: Could not connect to the server at {SERVER_URL}. {str(e)}")
        
        app = QtWidgets.QApplication(sys.argv)
        # Create a simple window to display the error without requiring user interaction
        error_window = QtWidgets.QWidget()
        error_window.setWindowTitle("Server Connection Error")
        error_window.setGeometry(300, 300, 500, 200)
        error_window.setStyleSheet("background-color: #2d2d2d; color: white;")
        
        layout = QVBoxLayout()
        error_label = QLabel(f"Could not connect to the server at {SERVER_URL}.\nPlease ensure the server is running and the URL is correct.")
        error_label.setStyleSheet("color: #ff6b6b; font-size: 14px; padding: 20px;")
        error_label.setWordWrap(True)
        layout.addWidget(error_label)
        
        error_window.setLayout(layout)
        error_window.show()
        
        # Auto close after 5 seconds
        QTimer.singleShot(5000, error_window.close)
        
        sys.exit(app.exec_())

    app = QtWidgets.QApplication(sys.argv)
    
    # Set application-wide stylesheet
    app.setStyleSheet("""
        QInputDialog {
            background-color: #2d2d2d;
            color: white;
        }
        QInputDialog QLabel {
            color: white;
        }
        QInputDialog QLineEdit {
            background-color: #3d3d3d;
            color: white;
            border: 1px solid #555;
            padding: 5px;
        }
        QFileDialog {
            background-color: #2d2d2d;
            color: white;
        }
        QFileDialog QLabel {
            color: white;
        }
        QFileDialog QLineEdit {
            background-color: #3d3d3d;
            color: white;
            border: 1px solid #555;
            padding: 5px;
        }
    """)
    
    window = FaceRecognitionClient()
    window.show()
    
    # Log application startup
    logging.info("Face Recognition Client started")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()