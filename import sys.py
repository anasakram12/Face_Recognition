import sys
import os
import cv2
import pickle
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from insightface.app import FaceAnalysis


class FaceRecognitionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            root='models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        self.embeddings_file = 'Database/face_embeddings.pkl'
        self.face_db = self.load_embeddings()

    def init_ui(self):
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 800, 400)

        layout = QHBoxLayout()

        # Left Side: Drag and Drop Area for Original Image
        self.original_image_label = QLabel("Drag and Drop Image Here")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFont(QFont("Arial", 12))
        self.original_image_label.setStyleSheet("border: 2px dashed #aaa; background-color: #f9f9f9;")
        self.original_image_label.setFixedSize(300, 300)
        self.original_image_label.setFrameShape(QFrame.Box)
        self.original_image_label.setAcceptDrops(True)
        self.original_image_label.dragEnterEvent = self.drag_enter_event
        self.original_image_label.dropEvent = self.drop_event

        # Right Side: Recognized Image Display
        self.recognized_image_label = QLabel("Recognized Image")
        self.recognized_image_label.setAlignment(Qt.AlignCenter)
        self.recognized_image_label.setFont(QFont("Arial", 12))
        self.recognized_image_label.setStyleSheet("border: 2px solid #aaa; background-color: #f9f9f9;")
        self.recognized_image_label.setFixedSize(300, 300)
        self.recognized_image_label.setFrameShape(QFrame.Box)

        # Match Details
        self.match_info_label = QLabel("Match Info: None")
        self.match_info_label.setFont(QFont("Arial", 10))
        self.match_info_label.setAlignment(Qt.AlignCenter)

        # Layout Arrangement
        layout.addWidget(self.original_image_label)
        layout.addWidget(self.recognized_image_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(self.match_info_label)

        self.setLayout(main_layout)

    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def drop_event(self, event):
        file_url = event.mimeData().urls()[0].toLocalFile()
        if file_url.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.display_original_image(file_url)
            self.recognize_face(file_url)
        else:
            QMessageBox.warning(self, "Invalid File", "Please drop a valid image file.")

    def display_original_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.original_image_label.setPixmap(pixmap.scaled(self.original_image_label.width(), self.original_image_label.height(), Qt.KeepAspectRatio))

    def load_embeddings(self):
        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                data = pickle.load(f)
                print(f"Loaded {len(data['embeddings'])} embeddings.")
                return data['embeddings']
        else:
            print("Embeddings file not found.")
            return {}

    def recognize_face(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load the image.")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.face_analyzer.get(img)

        if len(faces) == 0:
            self.match_info_label.setText("No face detected.")
            return

        query_embedding = faces[0].embedding

        # Find the most similar face
        best_match = None
        best_similarity = -1

        for identity, data in self.face_db.items():
            similarity = self.calculate_similarity(query_embedding, data['embedding'])
            if similarity > best_similarity:
                best_match = data
                best_similarity = similarity

        if best_match and best_similarity > 0.1:  # Threshold for match
            self.display_recognized_image(best_match['image_path'])
            self.match_info_label.setText(f"Match: {identity}, Confidence: {best_similarity:.2%}")
        else:
            self.recognized_image_label.setText("No Match Found")
            self.match_info_label.setText("No match found above the threshold.")

    def display_recognized_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.recognized_image_label.setPixmap(pixmap.scaled(self.recognized_image_label.width(), self.recognized_image_label.height(), Qt.KeepAspectRatio))

    @staticmethod
    def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(embedding1, embedding2))


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
