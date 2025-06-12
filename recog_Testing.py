# import sys
# import os
# import cv2
# import pickle
# import configparser
# import numpy as np
# from PyQt5 import QtWidgets, QtGui
# from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QFrame, QTableWidget, QTableWidgetItem, QMenuBar, QAction, QInputDialog, QHeaderView
# from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer, QTime, QDateTime, QMimeData,QUrl
# from PyQt5.QtGui import QPixmap, QFont, QDrag
# from insightface.app import FaceAnalysis
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

# class DraggableLabel(QLabel):
#     def __init__(self, file_path, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.file_path = file_path

#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             drag = QDrag(self)
#             mime_data = QMimeData()
#             mime_data.setUrls([QUrl.fromLocalFile(self.file_path)])
#             drag.setMimeData(mime_data)
#             drag.exec_(Qt.CopyAction | Qt.MoveAction)

# class FaceRecognitionApp(QtWidgets.QWidget):
#     def __init__(self):
#         super().__init__()
#         self.config_file = 'config.ini'
#         self.config = self.load_config()
#         self.threshold = float(self.config.get('Settings', 'threshold', fallback=0.5))
#         self.watch_folder = self.config.get('Settings', 'watch_folder', fallback=None)

#         self.init_ui()
#         self.face_analyzer = FaceAnalysis(
#             name='buffalo_l',
#             root='models',
#             providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
#         )
#         self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
#         self.embeddings_file = 'Database/face_embeddings.pkl'
#         self.face_db = self.load_embeddings()
#         self.observer = Observer()
#         self.recent_matches = []  # To store recent matches

#         if self.watch_folder:
#             self.start_watching_folder()

#     def init_ui(self):
#         self.setWindowTitle("Face Recognition System")
#         self.setGeometry(100, 100, 1600, 800)
#         self.setStyleSheet("""
#             QWidget {
#                 background-color: #1e1e1e;
#                 color: white;
#                 font-family: Arial, sans-serif;
#             }
#             QMenuBar {
#                 background-color: #2b2b2b;
#                 color: white;
#             }
#             QMenuBar::item:selected {
#                 background-color: #3d3d3d;
#             }
#             QTableWidget {
#                 background-color: #252526;
#                 gridline-color: #444;
#                 color: white;
#                 font-size: 14px;
#             }
#             QTableWidget QHeaderView::section {
#                 background-color: #3d3d3d;
#                 color: white;
#                 font-weight: bold;
#             }
#             QPushButton {
#                 background-color: #3a6ea5;
#                 color: white;
#                 border-radius: 5px;
#                 font-size: 14px;
#                 padding: 5px;
#             }
#             QPushButton:hover {
#                 background-color: #5591d2;
#             }
#             QLabel#timeLabel {
#                 font-size: 24px;
#                 font-weight: bold;
#                 color: #03DAC5;
#             }
#         """)

#         main_layout = QVBoxLayout()

#         # Menu Bar
#         self.menu_bar = QMenuBar(self)
#         file_menu = self.menu_bar.addMenu("File")

#         # Add Select Folder Action
#         select_folder_action = QAction("Select Folder", self)
#         select_folder_action.triggered.connect(self.select_folder)
#         file_menu.addAction(select_folder_action)

#         # Add Reload Pickle Action
#         reload_pickle_action = QAction("Reload Pickle", self)
#         reload_pickle_action.triggered.connect(self.reload_embeddings)
#         file_menu.addAction(reload_pickle_action)

#         # Add Set Threshold Action
#         set_threshold_action = QAction("Set Threshold", self)
#         set_threshold_action.triggered.connect(self.set_threshold)
#         file_menu.addAction(set_threshold_action)

#         # Create horizontal layout for logo and title
#         header_layout = QHBoxLayout()
        
#         # Logo Label
#         logo_label = QLabel()
#         logo_pixmap = QPixmap('logo.png')
#         scaled_logo = logo_pixmap.scaled(140, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         logo_label.setPixmap(scaled_logo)
#         logo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
#         # Title Label
#         title_label = QLabel("SmartFace Sentinel")
#         title_label.setStyleSheet("font-size: 34px; font-weight: bold;")
#         title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

#         # Add logo and title to horizontal layout
#         header_layout.addWidget(logo_label)
#         header_layout.addWidget(title_label)
#         header_layout.setAlignment(Qt.AlignCenter)
#         header_layout.setSpacing(10)

#         # Table for displaying last 10 inputs and outputs in 5 rows with details
#         self.result_table = QTableWidget(5, 6)
#         self.result_table.setHorizontalHeaderLabels(["Input Image ", "Recognized Image", "Details ", "Input Image ", "Recognized Image", "Details "])
#         self.result_table.horizontalHeader().setStretchLastSection(False)
#         self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
#         self.result_table.verticalHeader().setVisible(False)
#         self.result_table.setFixedHeight(600)
#         for row in range(5):
#             self.result_table.setRowHeight(row, 120)

#         # Styling the table
#         self.result_table.setStyleSheet("""
#             QTableWidget::item { padding: 10px; }
#         """)

#         # Time Label
#         self.time_label = QLabel()
#         self.time_label.setObjectName("timeLabel")
#         self.time_label.setAlignment(Qt.AlignBottom | Qt.AlignRight)
#         self.time_label.setStyleSheet("position: absolute; margin-right: 20px; margin-bottom: 20px;")

#         # Update the time every second
#         timer = QTimer(self)
#         timer.timeout.connect(self.update_time)
#         timer.start(1000)
#         self.update_time()

#         main_layout.setMenuBar(self.menu_bar)
#         main_layout.addLayout(header_layout)  # Add the header layout
#         main_layout.addWidget(self.result_table)
#         main_layout.addWidget(self.time_label)  # Add the time label at the bottom
#         self.setLayout(main_layout)

#     def update_time(self):
#         current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
#         self.time_label.setText(current_time)

#     def load_config(self):
#         config = configparser.ConfigParser()
#         if not os.path.exists(self.config_file):
#             config['Settings'] = {'threshold': '0.5', 'watch_folder': ''}
#             with open(self.config_file, 'w') as f:
#                 config.write(f)
#         else:
#             config.read(self.config_file)
#         return config

#     def save_config(self):
#         with open(self.config_file, 'w') as f:
#             self.config.write(f)

#     def set_threshold(self):
#         new_threshold, ok = QInputDialog.getDouble(self, "Set Threshold", "Enter new threshold (0.0 - 1.0):", self.threshold, 0.0, 1.0, 2)
#         if ok:
#             self.threshold = new_threshold
#             self.config.set('Settings', 'threshold', str(self.threshold))
#             self.save_config()
#             QMessageBox.information(self, "Threshold Updated", f"Threshold has been set to {self.threshold:.2f}")

#     def select_folder(self):
#         folder = QFileDialog.getExistingDirectory(self, "Select Folder to Watch")
#         if folder:
#             self.watch_folder = folder
#             self.config.set('Settings', 'watch_folder', self.watch_folder)
#             self.save_config()
#             self.start_watching_folder()

#     def start_watching_folder(self):
#         if self.watch_folder:
#             self.observer.stop()  # Stop any existing observers
#             self.observer = Observer()
#             event_handler = FolderWatchHandler(self)
#             self.observer.schedule(event_handler, self.watch_folder, recursive=False)
#             self.observer.start()
#             QMessageBox.information(self, "Folder Watching", f"Now watching folder: {self.watch_folder}")

#     def process_new_image(self, file_path):
#         timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
#         input_image_name = os.path.basename(file_path)
#         os.makedirs("Get/In", exist_ok=True)
#         os.makedirs("Get/Out", exist_ok=True)

#         # Save the input image
#         input_image_save_path = os.path.join("Get/In", f"{os.path.splitext(input_image_name)[0]}_{timestamp}.jpg")
#         cv2.imwrite(input_image_save_path, cv2.imread(file_path))

#         input_image_pixmap = QPixmap(file_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#         match_info, output_image_pixmap, output_filename, output_image_save_path, confidence = self.recognize_face(file_path, timestamp)

#         if output_image_save_path:
#             self.update_recent_matches(input_image_pixmap, output_image_pixmap, timestamp, output_filename, confidence, input_image_save_path)

#     def load_embeddings(self):
#         if os.path.exists(self.embeddings_file):
#             with open(self.embeddings_file, 'rb') as f:
#                 data = pickle.load(f)
#                 print(f"Loaded {len(data['embeddings'])} embeddings.")
#                 return data['embeddings']
#         else:
#             print("Embeddings file not found.")
#             return {}

#     def reload_embeddings(self):
#         self.face_db = self.load_embeddings()
#         QMessageBox.information(self, "Reload Pickle", "Embeddings file has been reloaded successfully.")

#     def recognize_face(self, file_path, timestamp):
#         img = cv2.imread(file_path)
#         if img is None:
#             QMessageBox.critical(self, "Error", "Failed to load the image.")
#             return "Error loading image", None, None, None, None

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faces = self.face_analyzer.get(img)

#         if len(faces) == 0:
#             unrecognized_image_path = os.path.join("Get/In", f"{os.path.splitext(os.path.basename(file_path))[0]}_unrecognized_{timestamp}.jpg")
#             cv2.imwrite(unrecognized_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#             return "No face detected", None, None, None, None

#         query_embedding = faces[0].embedding

#         # Find the most similar face
#         best_match = None
#         best_similarity = -1

#         for identity, data in self.face_db.items():
#             similarity = self.calculate_similarity(query_embedding, data['embedding'])
#             if similarity > best_similarity:
#                 best_match = data
#                 best_similarity = similarity

#         if best_match and best_similarity > self.threshold:  # Threshold for match
#             output_pixmap = QPixmap(best_match['image_path']).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
#             match_info = f"Confidence: {best_similarity:.2%}"
#             output_filename = os.path.basename(best_match['image_path'])
#             output_image_save_path = os.path.join("Get/Out", f"{os.path.splitext(output_filename)[0]}_{timestamp}.jpg")
#             cv2.imwrite(output_image_save_path, cv2.imread(best_match['image_path']))
#             return match_info, output_pixmap, output_filename, output_image_save_path, best_similarity
#         else:
#             unrecognized_image_path = os.path.join("Get/In", f"{os.path.splitext(os.path.basename(file_path))[0]}_unrecognized_{timestamp}.jpg")
#             cv2.imwrite(unrecognized_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#             return "No match found above the threshold", None, None, None, None

#     def update_recent_matches(self, input_image_pixmap, output_image_pixmap, timestamp, output_filename, confidence, input_file_path):
#         details = f"Timestamp: {timestamp}\nFile: {output_filename}"
#         if confidence is not None:
#             details += f"\nConfidence: {confidence:.2%}"

#         self.recent_matches.insert(0, (input_image_pixmap, output_image_pixmap, details, input_file_path))
#         if len(self.recent_matches) > 10:
#             self.recent_matches.pop()
#         self.refresh_table()

#     def refresh_table(self):
#         # Clear all rows
#         for i in range(5):
#             for j in range(6):
#                 self.result_table.setCellWidget(i, j, None)
#                 self.result_table.setItem(i, j, QTableWidgetItem(""))

#         # Add recent matches in the desired order
#         for row in range(5):
#             idx1 = row * 2
#             idx2 = idx1 + 1
#             if idx1 < len(self.recent_matches):
#                 input_pixmap_1, output_pixmap_1, details_1, file_path_1 = self.recent_matches[idx1]
#                 input_label_1 = DraggableLabel(file_path_1)
#                 input_label_1.setPixmap(input_pixmap_1)
#                 input_label_1.setAlignment(Qt.AlignCenter)
#                 self.result_table.setCellWidget(row, 0, input_label_1)

#                 output_label_1 = QLabel()
#                 if output_pixmap_1:
#                     output_label_1.setPixmap(output_pixmap_1)
#                     output_label_1.setAlignment(Qt.AlignCenter)
#                 self.result_table.setCellWidget(row, 1, output_label_1)

#                 self.result_table.setItem(row, 2, QTableWidgetItem(details_1))

#             if idx2 < len(self.recent_matches):
#                 input_pixmap_2, output_pixmap_2, details_2, file_path_2 = self.recent_matches[idx2]
#                 input_label_2 = DraggableLabel(file_path_2)
#                 input_label_2.setPixmap(input_pixmap_2)
#                 input_label_2.setAlignment(Qt.AlignCenter)
#                 self.result_table.setCellWidget(row, 3, input_label_2)

#                 output_label_2 = QLabel()
#                 if output_pixmap_2:
#                     output_label_2.setPixmap(output_pixmap_2)
#                     output_label_2.setAlignment(Qt.AlignCenter)
#                 self.result_table.setCellWidget(row, 4, output_label_2)

#                 self.result_table.setItem(row, 5, QTableWidgetItem(details_2))

#     @staticmethod
#     def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
#         embedding1 = embedding1 / np.linalg.norm(embedding1)
#         embedding2 = embedding2 / np.linalg.norm(embedding2)
#         return float(np.dot(embedding1, embedding2))

# class FolderWatchHandler(FileSystemEventHandler, QObject):
#     image_created_signal = pyqtSignal(str)

#     def __init__(self, app):
#         super().__init__()
#         self.app = app
#         self.image_created_signal.connect(self.app.process_new_image)

#     def on_created(self, event):
#         if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             self.image_created_signal.emit(event.src_path)

# def main():
#     app = QtWidgets.QApplication(sys.argv)
#     window = FaceRecognitionApp()
#     window.show()
#     sys.exit(app.exec_())

# if __name__ == "__main__":
#     main()

import sys
import os
import cv2
import pickle
import configparser
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QFrame, QTableWidget, QTableWidgetItem, QMenuBar, QAction, QInputDialog, QHeaderView ,QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer, QTime, QDateTime, QMimeData,QUrl
from PyQt5.QtGui import QPixmap, QFont, QDrag
from insightface.app import FaceAnalysis
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

class FaceRecognitionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.config_file = 'config.ini'
        self.config = self.load_config()
        self.threshold = float(self.config.get('Settings', 'threshold', fallback=0.5))
        self.watch_folder = self.config.get('Settings', 'watch_folder', fallback=None)
        self.path_label = None  # Will be initialized in init_ui

        self.init_ui()
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            root='models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        self.embeddings_file = 'Database/face_embeddings.pkl'
        self.face_db = self.load_embeddings()
        self.observer = Observer()
        self.recent_matches = []  # To store recent matches

        if self.watch_folder:
            self.start_watching_folder()

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
        reload_pickle_action = QAction("Reload Pickle", self)
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
        self.path_label.setAlignment(Qt.AlignLeft)
        # Table for displaying last 10 inputs and outputs in 5 rows with details
        self.result_table = QTableWidget(5, 6)
        self.result_table.setHorizontalHeaderLabels(["Input Image ", "Recognized Image", "Details ", "Input Image ", "Recognized Image", "Details "])
        self.result_table.horizontalHeader().setStretchLastSection(False)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setFixedHeight(600)
        for row in range(5):
            self.result_table.setRowHeight(row, 120)

        # Styling the table
        self.result_table.setStyleSheet("""
            QTableWidget::item { padding: 10px; }
        """)

        # Time Label
        self.time_label = QLabel()
        self.time_label.setObjectName("timeLabel")
        self.time_label.setAlignment(Qt.AlignBottom | Qt.AlignRight)
        self.time_label.setStyleSheet("position: absolute; margin-right: 20px; margin-bottom: 20px;")

        # Update the time every second
        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)
        self.update_time()

        main_layout.setMenuBar(self.menu_bar)
        main_layout.addLayout(header_layout)  # Add the header layout
        main_layout.addWidget(self.path_label) 
        main_layout.addWidget(self.result_table)      
        main_layout.addWidget(self.time_label)  # Add the time label at the bottom
        self.setLayout(main_layout)

    def update_time(self):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.time_label.setText(current_time)

    def load_config(self):
        config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            config['Settings'] = {'threshold': '0.5', 'watch_folder': ''}
            with open(self.config_file, 'w') as f:
                config.write(f)
        else:
            config.read(self.config_file)
        return config

    def save_config(self):
        with open(self.config_file, 'w') as f:
            self.config.write(f)

    def set_threshold(self):
        new_threshold, ok = QInputDialog.getDouble(self, "Set Threshold", "Enter new threshold (0.0 - 1.0):", self.threshold, 0.0, 1.0, 2)
        if ok:
            self.threshold = new_threshold
            self.config.set('Settings', 'threshold', str(self.threshold))
            self.save_config()
            QMessageBox.information(self, "Threshold Updated", f"Threshold has been set to {self.threshold:.2f}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Watch")
        if folder:
            self.watch_folder = folder
            self.config.set('Settings', 'watch_folder', self.watch_folder)
            self.save_config()
            self.start_watching_folder()
            self.path_label.setText(f"Watching: {self.watch_folder}")

    def start_watching_folder(self):
        if self.watch_folder:
            self.observer.stop()  # Stop any existing observers
            self.observer = Observer()
            event_handler = FolderWatchHandler(self)
            self.observer.schedule(event_handler, self.watch_folder, recursive=False)
            self.observer.start()
            QMessageBox.information(self, "Folder Watching", f"Now watching folder: {self.watch_folder}")

    def process_new_image(self, file_path):
        try:
            timestamp = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
            input_image_name = os.path.basename(file_path)
            os.makedirs("Get/In", exist_ok=True)
            os.makedirs("Get/Out", exist_ok=True)

            # Read input image with error handling
            input_image = cv2.imread(file_path)
            if input_image is None:
                print(f"Warning: Could not read image {file_path}")
                return

            # Save input image with error handling
            input_image_save_path = os.path.join("Get/In", f"{os.path.splitext(input_image_name)[0]}_{timestamp}.jpg")
            save_success = cv2.imwrite(input_image_save_path, input_image)
            
            if not save_success:
                print(f"Warning: Could not save image to {input_image_save_path}")
                return

            # Create pixmap with error handling
            input_image_pixmap = QPixmap(file_path)
            if input_image_pixmap.isNull():
                print(f"Warning: Could not create pixmap from {file_path}")
                return

            input_image_pixmap = input_image_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            match_info, output_image_pixmap, output_filename, output_image_save_path, confidence = self.recognize_face(file_path, timestamp)

            if output_image_save_path:
                self.update_recent_matches(input_image_pixmap, output_image_pixmap, timestamp, output_filename, confidence, input_image_save_path)

        except Exception as e:
            print(f"Warning: Error processing image {file_path}: {str(e)}")

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                print(f"Loaded {len(data['embeddings'])} embeddings.")
                return data['embeddings']
        else:
            print("Embeddings file not found.")
            return {}

    def reload_embeddings(self):
        self.face_db = self.load_embeddings()
        QMessageBox.information(self, "Reload Pickle", "Embeddings file has been reloaded successfully.")

    def recognize_face(self, file_path, timestamp):
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Warning: Could not read image {file_path}")
                return "Error loading image", "unrecognized", None, file_path, None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(img)

            if len(faces) == 0:
                unrecognized_image_path = os.path.join("Get/In", f"{os.path.splitext(os.path.basename(file_path))[0]}_unrecognized_{timestamp}.jpg")
                cv2.imwrite(unrecognized_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                return "No face detected", "unrecognized", None, unrecognized_image_path, None

            query_embedding = faces[0].embedding

            # Find the most similar face
            best_match = None
            best_similarity = -1

            for identity, data in self.face_db.items():
                similarity = self.calculate_similarity(query_embedding, data['embedding'])
                if similarity > best_similarity:
                    best_match = data
                    best_similarity = similarity

            if best_match and best_similarity > self.threshold:
                output_pixmap = QPixmap(best_match['image_path']).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                match_info = f"Confidence: {best_similarity:.2%}"
                output_filename = os.path.basename(best_match['image_path'])
                output_image_save_path = os.path.join("Get/Out", f"{os.path.splitext(output_filename)[0]}_{timestamp}.jpg")
                cv2.imwrite(output_image_save_path, cv2.imread(best_match['image_path']))
                return match_info, output_pixmap, output_filename, output_image_save_path, best_similarity
            else:
                unrecognized_image_path = os.path.join("Get/In", f"{os.path.splitext(os.path.basename(file_path))[0]}_unrecognized_{timestamp}.jpg")
                cv2.imwrite(unrecognized_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                return "No match found", "unrecognized", None, unrecognized_image_path, None
        except Exception as e:
            print(f"Warning: Error in face recognition for {file_path}: {str(e)}")
            return "Error in processing", "unrecognized", None, file_path, None
        


    def update_recent_matches(self, input_image_pixmap, output_image_pixmap, timestamp, output_filename, confidence, input_file_path):
        details = f"Timestamp: {timestamp}"
        if output_filename:
            details += f"\nFile: {output_filename}"
        if confidence is not None:
            details += f"\nConfidence: {confidence:.2%}"
        else:
            details += "\nStatus: Unrecognized"

        self.recent_matches.insert(0, (input_image_pixmap, output_image_pixmap, details, input_file_path))
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
                        self.result_table.setItem(row, column_start + 2, QTableWidgetItem(details))

        except Exception as e:
            print(f"Warning: Error refreshing table: {str(e)}")

    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(embedding1, embedding2))

class FolderWatchHandler(FileSystemEventHandler, QObject):
    image_created_signal = pyqtSignal(str)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.image_created_signal.connect(self.app.process_new_image)

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.image_created_signal.emit(event.src_path)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
