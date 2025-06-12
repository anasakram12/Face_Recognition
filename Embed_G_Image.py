import os
import cv2
import pickle
import datetime
import shutil
from PySide6 import QtWidgets
from PySide6.QtWidgets import (QFileDialog, QMessageBox, QLabel, QVBoxLayout, 
                           QPushButton, QProgressBar, QLineEdit, QHBoxLayout, 
                           QFrame, QWidget, QScrollArea)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor
from insightface.app import FaceAnalysis

class CustomButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setFixedHeight(30)  # Reduced height
        self.setFont(QFont("Segoe UI", 9))  # Smaller font
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2d5a88;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #3d7ab8;
            }
            QPushButton:pressed {
                background-color: #1d3a58;
            }
            QPushButton:disabled {
                background-color: #1a1a1a;
                color: #666666;
            }
        """)

class FaceEmbeddingApp(QWidget):
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
        
        self.embeddings_file = os.path.join(self.database_dir, 'face_embeddings_mod.pkl')
        self.images_dir = os.path.join(self.database_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.face_db = {}
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
        """)

    def init_ui(self):
        self.setWindowTitle("Embedding Gen Group")
        self.setFixedSize(600, 400)  # Fixed compact size
        self.setWindowIcon(QIcon('icon.png'))
        
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)  # Reduced spacing
        
        # Left Panel
        left_panel = QWidget()
        left_panel.setStyleSheet("""
            QWidget {
                background-color: #252526;
                border-radius: 5px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)  # Reduced margins
        left_layout.setSpacing(8)  # Reduced spacing

        # Title
        title_label = QLabel("Embedding Gen Group")
        title_label.setFont(QFont("Segoe UI", 12, QFont.Bold))  # Smaller font
        title_label.setStyleSheet("color: #ffffff; margin-bottom: 10px;")
        
        # Input Section
        self.input_folder_label = QLabel("Input Folder: Not Selected")
        self.input_folder_label.setFont(QFont("Segoe UI", 9))
        self.input_folder_label.setStyleSheet("color: #cccccc;")
        
        self.select_folder_button = CustomButton("Select Input Folder")
        self.select_folder_button.clicked.connect(self.select_input_folder)
        
        # Progress Section
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(15)  # Smaller progress bar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2d5a88;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2d5a88;
            }
        """)
        
        self.start_button = CustomButton("Start Embedding Generation")
        self.start_button.clicked.connect(self.start_embedding)
        
        # Stats Section
        stats_frame = QFrame()
        stats_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        
        self.stats_label = QLabel("Statistics\n\nProcessed: 0\nFaces: 0\nSkipped: 0")
        self.stats_label.setFont(QFont("Segoe UI", 9))
        self.stats_label.setStyleSheet("color: #cccccc;")
        stats_layout.addWidget(self.stats_label)

        # Add widgets to left panel
        left_layout.addWidget(title_label)
        left_layout.addWidget(self.input_folder_label)
        left_layout.addWidget(self.select_folder_button)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.start_button)
        left_layout.addWidget(stats_frame)
        left_layout.addStretch()

        # Right Panel (Image Display)
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QWidget {
                background-color: #252526;
                border-radius: 5px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        self.processing_label = QLabel("Processing: None")
        self.processing_label.setFont(QFont("Segoe UI", 9))
        self.processing_label.setStyleSheet("color: #cccccc;")

        # Create a scroll area for the image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #2d5a88;
                border-radius: 4px;
                background-color: #1e1e1e;
            }
            QScrollBar {
                width: 10px;
                background: #1e1e1e;
            }
            QScrollBar::handle {
                background-color: #2d5a88;
                border-radius: 5px;
            }
        """)

        self.image_display_label = QLabel()
        self.image_display_label.setMinimumSize(QSize(250, 250))  # Reduced minimum size
        self.image_display_label.setAlignment(Qt.AlignCenter)
        scroll_area.setWidget(self.image_display_label)

        right_layout.addWidget(self.processing_label)
        right_layout.addWidget(scroll_area)

        # Add panels to main layout with adjusted proportions
        main_layout.addWidget(left_panel, 45)  # 45% width
        main_layout.addWidget(right_panel, 55)  # 55% width
        
        self.setLayout(main_layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder = folder
            self.input_folder_label.setText(f"Input: {os.path.basename(folder)}")
            self.update_stats()

    def update_stats(self):
        if hasattr(self, 'input_folder'):
            image_count = len([f for f in os.listdir(self.input_folder) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            self.stats_label.setText(f"Statistics\n\n"
                                   f"Images: {image_count}\n"
                                   f"Processed: {len(self.face_db)}\n"
                                   f"Skipped: 0")

    def start_embedding(self):
        if not hasattr(self, 'input_folder'):
            QMessageBox.critical(self, "Error", "Please select an input folder.")
            return

        try:
            self.generate_embeddings(self.input_folder)
            QMessageBox.information(self, "Success", "Processing completed!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def calculate_face_area_percentage(self, bbox, image_shape):
        """Calculate what percentage of the image area the face bounding box occupies"""
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        face_area = face_width * face_height
        
        image_height, image_width = image_shape[:2]
        image_area = image_width * image_height
        
        return (face_area / image_area) * 100

    def generate_embeddings(self, folder_path):
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder {folder_path} does not exist!")

        # Clear existing embeddings for regeneration
        self.face_db = {}
        
        processed_files = 0
        faces_detected = 0
        skipped_files = 0

        files = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.progress_bar.setMaximum(len(files))

        for i, filename in enumerate(files):
            self.progress_bar.setValue(i + 1)
            self.processing_label.setText(f"Processing: {filename}")
            QtWidgets.QApplication.processEvents()

            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                skipped_files += 1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(img)

            if faces:
                # Find the face with the largest bounding box area
                largest_face = None
                largest_area = 0
                
                for face in faces:
                    bbox = face.bbox
                    face_width = bbox[2] - bbox[0]
                    face_height = bbox[3] - bbox[1]
                    face_area = face_width * face_height
                    
                    if face_area > largest_area:
                        largest_area = face_area
                        largest_face = face
                
                if largest_face is not None:
                    # Check if the face area is at least 20% of the image
                    area_percentage = self.calculate_face_area_percentage(largest_face.bbox, img.shape)
                    
                    if area_percentage >= 20.0:
                        identity = os.path.splitext(filename)[0]
                        
                        # Handle case where input and output folders might be the same
                        new_image_path = os.path.join(self.images_dir, filename)
                        
                        # Use absolute paths for accurate comparison
                        abs_img_path = os.path.abspath(img_path)
                        abs_new_path = os.path.abspath(new_image_path)
                        
                        if abs_img_path != abs_new_path:  # Only copy if different absolute paths
                            try:
                                shutil.copy2(img_path, new_image_path)
                            except shutil.SameFileError:
                                # Files are the same, use original path
                                new_image_path = img_path
                        else:
                            new_image_path = img_path  # Use original path if same folder
                        
                        self.face_db[identity] = {
                            'embedding': largest_face.embedding,
                            'image_path': new_image_path,
                            'processed_date': datetime.datetime.now(),
                            'face_area_percentage': area_percentage
                        }

                        # Display the processed image
                        height, width, channel = img.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(img.data, width, height, bytes_per_line, 
                                    QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_image)
                        
                        # Scale the image to fit the display area (250x250)
                        scaled_pixmap = pixmap.scaled(250, 250,
                                                    Qt.KeepAspectRatio,
                                                    Qt.SmoothTransformation)
                        self.image_display_label.setPixmap(scaled_pixmap)

                        processed_files += 1
                        faces_detected += 1
                    else:
                        # Face is too small (less than 20% of image)
                        skipped_files += 1
                        print(f"Skipped {filename}: Face area {area_percentage:.1f}% < 20%")
                else:
                    skipped_files += 1
            else:
                # No faces detected
                skipped_files += 1
                
            # Update statistics
            self.stats_label.setText(f"Statistics\n\n"
                                f"Processed: {processed_files}\n"
                                f"Faces: {faces_detected}\n"
                                f"Skipped: {skipped_files}")

        self.save_embeddings()
        self.progress_bar.setValue(len(files))
        self.processing_label.setText(f"Completed - {processed_files} processed, {skipped_files} skipped")
        print(f"Processing completed: {processed_files} faces processed, {skipped_files} images skipped")

    def save_embeddings(self):
        try:
            # For regeneration, we replace the entire file rather than merging
            # This ensures clean regeneration of embeddings
            with open(self.embeddings_file, 'wb') as f:
                data = {
                    'embeddings': self.face_db,
                    'timestamp': datetime.datetime.now(),
                    'total_faces': len(self.face_db)
                }
                pickle.dump(data, f)
            
            print(f"Embeddings saved to: {self.embeddings_file}")
            print(f"Total embeddings: {len(self.face_db)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving embeddings: {str(e)}")

def main():
    app = QtWidgets.QApplication([])
    window = FaceEmbeddingApp()
    window.show()
    app.exec()  # Note: In PySide6, exec() doesn't need underscore

if __name__ == "__main__":
    main()