import os
import cv2
import pickle
import datetime
import shutil
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel, QVBoxLayout, QPushButton, QProgressBar, QLineEdit, QHBoxLayout, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from insightface.app import FaceAnalysis

class FaceEmbeddingApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            root='models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        # Create Database directory if it doesn't exist
        self.database_dir = 'Database'
        os.makedirs(self.database_dir, exist_ok=True)
        
        self.embeddings_file = os.path.join(self.database_dir, 'face_embeddings.pkl')
        self.images_dir = os.path.join(self.database_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.face_db = {}

    def init_ui(self):
        self.setWindowTitle("Face Embedding Generator")
        self.setGeometry(100, 100, 800, 400)
        self.setWindowIcon(QIcon('icon.png'))

        layout = QHBoxLayout()

        # Left Layout (Controls)
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(15)

        self.input_folder_label = QLabel("Input Folder: Not Selected")
        self.input_folder_label.setFont(QFont("Arial", 10))
        self.input_folder_label.setStyleSheet("color: #333;")

        self.select_folder_button = QPushButton("Select Input Folder")
        self.select_folder_button.setFont(QFont("Arial", 12))
        self.select_folder_button.clicked.connect(self.select_input_folder)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")

        self.start_button = QPushButton("Start Embedding Generation")
        self.start_button.setFont(QFont("Arial", 12))
        self.start_button.clicked.connect(self.start_embedding)

        controls_layout.addWidget(self.input_folder_label)
        controls_layout.addWidget(self.select_folder_button)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.start_button)

        # Right Layout (Image Display)
        self.processing_label = QLabel("Processing: None")
        self.processing_label.setFont(QFont("Arial", 10))

        self.image_display_label = QLabel()
        self.image_display_label.setFixedSize(300, 300)
        self.image_display_label.setFrameShape(QFrame.Box)
        self.image_display_label.setStyleSheet("border: 1px solid #ddd; background-color: #f9f9f9;")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.processing_label)
        right_layout.addWidget(self.image_display_label)

        layout.addLayout(controls_layout, stretch=2)
        layout.addLayout(right_layout, stretch=1)

        self.setLayout(layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder = folder
            self.input_folder_label.setText(f"Input Folder: {folder}")

    def start_embedding(self):
        if not hasattr(self, 'input_folder'):
            QMessageBox.critical(self, "Error", "Please select an input folder.")
            return

        try:
            self.generate_embeddings(self.input_folder)
            QMessageBox.information(self, "Completed", "Embeddings and images have been successfully saved to the Database folder!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def generate_embeddings(self, folder_path):
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder {folder_path} does not exist!")

        processed_files = 0
        self.face_db = {}

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.progress_bar.setMaximum(len(files))

        for i, filename in enumerate(files):
            self.progress_bar.setValue(i + 1)
            self.processing_label.setText(f"Processing: {filename}")
            QtWidgets.QApplication.processEvents()

            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Could not read image: {filename}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(img)

            if len(faces) > 0:
                face = faces[0]
                identity = os.path.splitext(filename)[0]
                
                # Save the image to the Database/images directory
                new_image_path = os.path.join(self.images_dir, filename)
                shutil.copy2(img_path, new_image_path)
                
                self.face_db[identity] = {
                    'embedding': face.embedding,
                    'image_path': new_image_path,  # Store the new path in Database/images
                    'processed_date': datetime.datetime.now()
                }

                # Display the processed face
                face_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channel = face_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(face_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_display_label.setPixmap(pixmap.scaled(self.image_display_label.width(), self.image_display_label.height(), Qt.KeepAspectRatio))

                processed_files += 1

        self.save_embeddings()
        self.progress_bar.setValue(len(files))
        self.processing_label.setText("Processing: Completed")
        print(f"Processed {processed_files} files.")
        print(f"Images and embeddings saved to {self.database_dir}")

    def save_embeddings(self):
        try:
            # Load existing data if the pickle file exists
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    existing_data = pickle.load(f)
                existing_embeddings = existing_data.get('embeddings', {})
            else:
                existing_embeddings = {}

            # Update the embeddings with the new ones
            existing_embeddings.update(self.face_db)

            # Save the updated data back to the pickle file
            with open(self.embeddings_file, 'wb') as f:
                data = {
                    'embeddings': existing_embeddings,
                    'timestamp': datetime.datetime.now()
                }
                pickle.dump(data, f)
            print(f"Embeddings updated and saved to {self.embeddings_file}")
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")


# Main Application
def main():
    app = QtWidgets.QApplication([])
    window = FaceEmbeddingApp()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()