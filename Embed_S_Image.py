# Add these at the very top of the file
import os
import cv2
import pickle
import datetime
import shutil
# Add the missing insightface import
import insightface
from insightface.app import FaceAnalysis
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QFileDialog, QMessageBox, QLabel, QVBoxLayout, 
                           QPushButton, QLineEdit, QHBoxLayout, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QDragEnterEvent, QDropEvent, QPalette, QColor

class DragDropLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("\n Drop Image Here \n or \n Click to Browse")  # Removed extra newlines
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #666;
                border-radius: 6px;  # Reduced from 8px
                background-color: #2a2a2a;
                padding: 10px;  # Reduced from 20px
                color: #aaa;
                font-size: 12px;  # Reduced from 14px
            }
            QLabel:hover {
                border-color: #888;
                background-color: #303030;
            }
        """)
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasImage or event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            self.parent().process_dropped_image(url.toLocalFile())
            event.accept()
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent().select_image()

class SingleFaceEmbeddingApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            root='models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        self.database_dir = 'Database'
        os.makedirs(self.database_dir, exist_ok=True)
        
        self.embeddings_file = os.path.join(self.database_dir, 'face_embeddings.pkl')
        self.images_dir = os.path.join(self.database_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)

    def init_ui(self):
        self.setWindowTitle("Face Embedding Generator")
        self.setGeometry(100, 100, 300, 400)  # Reduced from 400x400
        self.setWindowIcon(QIcon('icon.png'))
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #2d2d2d;
                color: #ffffff;
                selection-background-color: #3d3d3d;
            }
            QLineEdit:focus {
                border: 1px solid #666;
                background-color: #333;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3880;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
        """)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)  # Reduced from 25
        main_layout.setContentsMargins(20, 20, 20, 20)  # Reduced from 30
        
        # Title
        title_label = QLabel("Face Embedding Generator")
        title_label.setStyleSheet("""
            font-size: 18px;  # Reduced from 24px
            font-weight: bold;
            color: #fff;
            margin-bottom: 10px;  # Reduced from 20px
        """)
        title_label.setAlignment(Qt.AlignCenter)
        
        # Drag & Drop Area
        self.drop_label = DragDropLabel()
        self.drop_label.setFixedSize(260, 180)  # Reduced from 400x300
        
        # Input Frame
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 6px;  # Reduced from 8px
                padding: 10px;  # Reduced from 15px
            }
        """)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setSpacing(8)  # Added spacing control
        
        # Filename Input
        filename_layout = QHBoxLayout()
        self.filename_label = QLabel("Name:")  # Shortened label
        self.filename_label.setFont(QFont("Arial", 10))  # Reduced from 11
        self.filename_input = QLineEdit()
        self.filename_input.setPlaceholderText("Enter identity name...")
        self.filename_input.setFont(QFont("Arial", 10))  # Reduced from 11
        self.filename_input.setMinimumHeight(25)  # Reduced from 35
        filename_layout.addWidget(self.filename_label)
        filename_layout.addWidget(self.filename_input)
        
        # Save Button
        self.save_button = QPushButton("Save Embedding")
        self.save_button.setFont(QFont("Arial", 10))  # Reduced from 12
        self.save_button.setMinimumHeight(30)  # Reduced from 40
        self.save_button.clicked.connect(self.save_embedding)
        self.save_button.setEnabled(False)
        
        # Status Label
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Arial", 9))  # Reduced from 10
        self.status_label.setStyleSheet("""
            color: #aaa;
            padding: 5px;  # Reduced from 10px
            background-color: #2a2a2a;
            border-radius: 4px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.drop_label, alignment=Qt.AlignCenter)
        input_layout.addLayout(filename_layout)
        input_layout.addWidget(self.save_button)
        main_layout.addWidget(input_frame)
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
        
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.process_dropped_image(file_path)
            
    def process_dropped_image(self, file_path):
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.show_error_message("Invalid File", "Please select an image file.")
            return
            
        self.current_image_path = file_path
        pixmap = QPixmap(file_path)
        self.drop_label.setPixmap(pixmap.scaled(
            self.drop_label.width(), 
            self.drop_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        self.save_button.setEnabled(True)
        self.update_status("Image loaded. Enter identity name and click Save.", "info")
        
    def update_status(self, message, status_type="info"):
        color = "#4caf50" if status_type == "success" else "#2196f3" if status_type == "info" else "#f44336"
        self.status_label.setStyleSheet(f"""
            color: {color};
            padding: 10px;
            background-color: #2a2a2a;
            border-radius: 4px;
            font-weight: bold;
        """)
        self.status_label.setText(message)
        
    def show_error_message(self, title, message):
        error_box = QMessageBox(self)
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setStyleSheet("""
            QMessageBox {
                background-color: #2a2a2a;
                color: #ffffff;
            }
            QMessageBox QPushButton {
                padding: 5px 20px;
                background-color: #0d47a1;
                border: none;
                border-radius: 3px;
                color: white;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background-color: #1565c0;
            }
        """)
        error_box.exec_()
            
    def save_embedding(self):
        if not hasattr(self, 'current_image_path'):
            self.show_error_message("Error", "Please select an image first.")
            return
            
        identity = self.filename_input.text().strip()
        if not identity:
            self.show_error_message("Error", "Please enter an identity name.")
            return
            
        try:
            img = cv2.imread(self.current_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(img)
            
            if not faces:
                self.show_error_message("Error", "No face detected in the image.")
                return
                
            face = faces[0]
            new_image_name = f"{identity}{os.path.splitext(self.current_image_path)[1]}"
            new_image_path = os.path.join(self.images_dir, new_image_name)
            shutil.copy2(self.current_image_path, new_image_path)
            
            new_embedding = {
                identity: {
                    'embedding': face.embedding,
                    'image_path': new_image_path,
                    'processed_date': datetime.datetime.now()
                }
            }
            
            existing_embeddings = {}
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    existing_data = pickle.load(f)
                    existing_embeddings = existing_data.get('embeddings', {})
                    
            existing_embeddings.update(new_embedding)
            with open(self.embeddings_file, 'wb') as f:
                data = {
                    'embeddings': existing_embeddings,
                    'timestamp': datetime.datetime.now()
                }
                pickle.dump(data, f)
                
            self.update_status("Successfully saved embedding and image!", "success")
            self.filename_input.clear()
            self.drop_label.setText("\n\n Drop Image Here \n or \n Click to Browse")
            self.save_button.setEnabled(False)
            
        except Exception as e:
            self.show_error_message("Error", f"Error saving embedding: {str(e)}")

def main():
    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    
    # Set dark theme palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = SingleFaceEmbeddingApp()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()