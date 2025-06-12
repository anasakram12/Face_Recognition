import sys
import os
import cv2
import pickle
import configparser
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, 
                           QFrame, QTableWidget, QTableWidgetItem, QMenuBar, QAction, 
                           QInputDialog, QHeaderView, QProgressBar, QPushButton, QWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer, QDateTime, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QFont, QColor, QPainter, QPainterPath, QLinearGradient
from insightface.app import FaceAnalysis
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class RoundedPhotoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.setMinimumSize(120, 120)

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            path = QPainterPath()
            path.addEllipse(0, 0, self.width(), self.height())
            
            painter.setClipPath(path)
            scaled_pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)

            pen = painter.pen()
            pen.setWidth(2)
            pen.setColor(QColor("#03DAC5"))
            painter.setPen(pen)
            painter.drawEllipse(1, 1, self.width()-2, self.height()-2)

class AnimatedProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #03DAC5;
                border-radius: 5px;
                text-align: center;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #03DAC5;
                border-radius: 3px;
            }
        """)
        self.setTextVisible(False)
        self.animation = QPropertyAnimation(self, b"value")
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.setDuration(1000)

    def animateValue(self, value):
        self.animation.setStartValue(self.value())
        self.animation.setEndValue(value)
        self.animation.start()

class ModernButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            ModernButton {
                background-color: #03DAC5;
                color: #1e1e1e;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            ModernButton:hover {
                background-color: #018786;
            }
            ModernButton:pressed {
                background-color: #015958;
            }
        """)

class FaceRecognitionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.config_file = 'config_front.ini'
        self.config = self.load_config()
        self.threshold = float(self.config.get('Settings', 'threshold', fallback=0.5))
        self.watch_folder = self.config.get('Settings', 'watch_folder', fallback=None)
        self.recent_matches = []
        
        # Initialize UI components
        self.init_ui()
        
        # Initialize face recognition
        self.init_face_recognition()
        
        # Start folder watching if configured
        self.observer = Observer()
        if self.watch_folder:
            self.start_watching_folder()

    def init_ui(self):
        # Create status bar first
        status_bar = self.create_status_bar()
        
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Add components
        header_layout = self.create_header()
        control_panel = self.create_control_panel()
        self.result_table = self.create_results_table()
        
        main_layout.addLayout(header_layout)
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.result_table)
        main_layout.addWidget(status_bar)
        
        self.setLayout(main_layout)
        
        # Window setup
        self.setWindowTitle("SmartFace Sentinel - Advanced Face Recognition System")
        self.setGeometry(100, 100, 1800, 1000)
        self.apply_styles()

    def init_face_recognition(self):
        try:
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                root='models',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            self.embeddings_file = 'Database/face_embeddings.pkl'
            self.face_db = self.load_embeddings()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize face recognition: {str(e)}")
            sys.exit(1)

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
        try:
            with open(self.config_file, 'w') as f:
                self.config.write(f)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to save configuration: {str(e)}")

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QMenuBar {
                background-color: #1e1e1e;
                color: #ffffff;
                border-bottom: 2px solid #03DAC5;
            }
            QMenuBar::item:selected {
                background-color: #03DAC5;
                color: #1e1e1e;
            }
            QTableWidget {
                background-color: #1e1e1e;
                gridline-color: #03DAC5;
                border: 2px solid #03DAC5;
                border-radius: 10px;
            }
            QTableWidget QHeaderView::section {
                background-color: #03DAC5;
                color: #1e1e1e;
                font-weight: bold;
                padding: 8px;
                border: none;
            }
            QLabel#statusLabel {
                color: #03DAC5;
                font-size: 16px;
                font-weight: bold;
            }
        """)

    def create_header(self):
        header_layout = QHBoxLayout()
        
        # Logo
        logo_label = QLabel()
        if os.path.exists('logo.png'):
            logo_pixmap = QPixmap('logo.png')
            scaled_logo = logo_pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
        else:
            logo_label.setText("Logo not found")
            logo_label.setStyleSheet("color: #03DAC5;")
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        title_label = QLabel("SmartFace Sentinel")
        title_label.setStyleSheet("font-size: 48px; font-weight: bold; color: #03DAC5;")
        subtitle_label = QLabel("Advanced Face Recognition System")
        subtitle_label.setStyleSheet("font-size: 24px; color: #BB86FC;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        header_layout.addWidget(logo_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        return header_layout

    def create_control_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 2px solid #03DAC5;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        layout = QHBoxLayout(panel)
        
        folder_btn = ModernButton("Select Watch Folder")
        folder_btn.clicked.connect(self.select_folder)
        
        threshold_btn = ModernButton("Set Threshold")
        threshold_btn.clicked.connect(self.set_threshold)
        
        reload_btn = ModernButton("Reload Database")
        reload_btn.clicked.connect(self.reload_embeddings)
        
        self.progress_bar = AnimatedProgressBar()
        self.progress_bar.setFixedHeight(10)
        
        layout.addWidget(folder_btn)
        layout.addWidget(threshold_btn)
        layout.addWidget(reload_btn)
        layout.addWidget(self.progress_bar)
        
        return panel

    def create_results_table(self):
        table = QTableWidget(5, 6)
        table.setHorizontalHeaderLabels([
            "Input Image", "Match Result", "Details",
            "Input Image", "Match Result", "Details"
        ])
        
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(True)
        table.setFrameShape(QFrame.NoFrame)
        
        for row in range(5):
            table.setRowHeight(row, 150)
        
        return table

    def create_status_bar(self):
        status_bar = QFrame()
        status_bar.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 2px solid #03DAC5;
                border-radius: 10px;
                padding: 5px;
            }
        """)
        
        layout = QHBoxLayout(status_bar)
        
        self.status_label = QLabel()
        self.status_label.setObjectName("statusLabel")
        
        self.time_label = QLabel()
        self.time_label.setStyleSheet("font-size: 20px; color: #03DAC5;")
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.time_label)
        
        # Update time every second
        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)
        self.update_time()
        
        return status_bar

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Watch")
        if folder:
            self.watch_folder = folder
            self.config.set('Settings', 'watch_folder', self.watch_folder)
            self.save_config()
            self.start_watching_folder()
            self.status_label.setText(f"Watching folder: {folder}")

    def set_threshold(self):
        new_threshold, ok = QInputDialog.getDouble(
            self, "Set Threshold", 
            "Enter new threshold (0.0 - 1.0):", 
            self.threshold, 0.0, 1.0, 2
        )
        if ok:
            self.threshold = new_threshold
            self.config.set('Settings', 'threshold', str(self.threshold))
            self.save_config()
            self.status_label.setText(f"Threshold set to: {self.threshold:.2f}")

    def load_embeddings(self):
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    if hasattr(self, 'status_label'):
                        self.status_label.setText(f"Loaded {len(data['embeddings'])} embeddings")
                    return data['embeddings']
            else:
                if hasattr(self, 'status_label'):
                    self.status_label.setText("Embeddings file not found")
                return {}
        except Exception as e:
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Error loading embeddings: {str(e)}")
            return {}

    def reload_embeddings(self):
        self.face_db = self.load_embeddings()
        self.status_label.setText("Database reloaded successfully")

    def start_watching_folder(self):
        if self.watch_folder:
            try:
                self.observer.stop()
                self.observer = Observer()
                event_handler = FolderWatchHandler(self)
                self.observer.schedule(event_handler, self.watch_folder, recursive=False)
                self.observer.start()
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to start folder watching: {str(e)}")

    def recognize_face(self, file_path):
        try:
            img = cv2.imread(file_path)
            if img is None:
                self.status_label.setText("Error loading image")
                return "Error loading image", None, None, None, None

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(img)
            timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")

            if len(faces) == 0:
                self.status_label.setText("No face detected")
                return "No face detected", None, None, timestamp, None

            query_embedding = faces[0].embedding
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
                self.status_label.setText(f"Match found: {output_filename}")
                return match_info, output_pixmap, output_filename, timestamp, best_similarity
            else:
                self.status_label.setText("No match found above threshold")
                return "No match found above threshold", None, None, timestamp, None
        except Exception as e:
            self.status_label.setText(f"Error during face recognition: {str(e)}")
            return f"Error: {str(e)}", None, None, QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss"), None

    def process_new_image(self, file_path):
        try:
            self.progress_bar.animateValue(0)
            self.status_label.setText(f"Processing: {os.path.basename(file_path)}")
            
            self.progress_bar.animateValue(30)
            input_image_pixmap = QPixmap(file_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.progress_bar.animateValue(60)
            match_info, output_image_pixmap, output_filename, timestamp, confidence = self.recognize_face(file_path)
            
            self.progress_bar.animateValue(90)
            self.update_recent_matches(input_image_pixmap, output_image_pixmap, timestamp, output_filename, confidence)
            
            self.progress_bar.animateValue(100)
            QTimer.singleShot(1000, lambda: self.progress_bar.animateValue(0))
        except Exception as e:
            self.status_label.setText(f"Error processing image: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to process image: {str(e)}")

    def update_time(self):
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.time_label.setText(current_time)

    def update_recent_matches(self, input_image_pixmap, output_image_pixmap, timestamp, output_filename, confidence):
        try:
            details = f"Timestamp: {timestamp}"
            if output_filename:
                details += f"\nFile: {output_filename}"
            if confidence is not None:
                details += f"\nConfidence: {confidence:.2%}"
                details += f"\nStatus: {'Match' if confidence > self.threshold else 'No Match'}"
            
            self.recent_matches.insert(0, (input_image_pixmap, output_image_pixmap, details))
            if len(self.recent_matches) > 10:
                self.recent_matches.pop()
            
            self.refresh_table()
        except Exception as e:
            self.status_label.setText(f"Error updating matches: {str(e)}")

    def refresh_table(self):
        try:
            for i in range(5):
                for j in range(6):
                    self.result_table.setCellWidget(i, j, None)
                    self.result_table.setItem(i, j, None)

            for row in range(5):
                idx1, idx2 = row * 2, row * 2 + 1
                
                for idx, col_offset in [(idx1, 0), (idx2, 3)]:
                    if idx < len(self.recent_matches):
                        input_pixmap, output_pixmap, details = self.recent_matches[idx]
                        
                        if input_pixmap:
                            input_widget = RoundedPhotoWidget()
                            input_widget.setPixmap(input_pixmap)
                            self.result_table.setCellWidget(row, col_offset, input_widget)
                        
                        if output_pixmap:
                            output_widget = RoundedPhotoWidget()
                            output_widget.setPixmap(output_pixmap)
                            self.result_table.setCellWidget(row, col_offset + 1, output_widget)
                        
                        details_item = QTableWidgetItem(details)
                        details_item.setTextAlignment(Qt.AlignCenter)
                        self.result_table.setItem(row, col_offset + 2, details_item)
        except Exception as e:
            self.status_label.setText(f"Error refreshing table: {str(e)}")

    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        try:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            return float(np.dot(embedding1, embedding2))
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return -1.0

class FolderWatchHandler(FileSystemEventHandler, QObject):
    image_created_signal = pyqtSignal(str)

    def __init__(self, app):
        super().__init__()
        QObject.__init__(self)
        self.app = app
        self.image_created_signal.connect(self.app.process_new_image, Qt.QueuedConnection)

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.image_created_signal.emit(event.src_path)

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        font = QFont("Segoe UI", 10)
        app.setFont(font)
        app.setStyle("Fusion")
        
        window = FaceRecognitionApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Critical Error", f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
