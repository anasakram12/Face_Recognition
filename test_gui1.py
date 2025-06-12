import os
import cv2
import pickle
import shutil
from datetime import datetime
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLineEdit, QLabel, QGridLayout,
                           QScrollArea, QMessageBox, QFileDialog, QProgressDialog,
                           QInputDialog, QComboBox, QFrame)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon

class LoadDatabaseThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(dict)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        try:
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
                self.finished.emit(data.get('embeddings', {}))
        except Exception as e:
            self.finished.emit({})

class ImageLoader(QThread):
    image_loaded = pyqtSignal(str, QPixmap)
    
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        
    def run(self):
        try:
            img = cv2.imread(self.image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_loaded.emit(self.image_path, scaled_pixmap)
        except Exception:
            pass

class ScrollContainer(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.NoFrame)
        
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.scroll_area)
        
        # Enhanced scrollbar styling
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QScrollBar:vertical {
                border: none;
                background: #2d2d2d;
                width: 14px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #4a4a4a;
                min-height: 30px;
                border-radius: 7px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #5a5a5a;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

class CustomButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setFixedHeight(30)
        self.setFont(QFont("Segoe UI", 9))
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
                background-color: #666666;
            }
        """)

class LazyImageCard(QWidget):
    id_changed = pyqtSignal(str, str)  # old_id, new_id
    image_replaced = pyqtSignal(str)    # id
    entry_deleted = pyqtSignal(str)     # id
    file_renamed = pyqtSignal(str, str) # id, new_name
    
    def __init__(self, id_name, image_path, parent=None):
        super().__init__(parent)
        self.id_name = id_name
        self.image_path = image_path
        self.image_loader = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setFixedSize(180, 180)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # ID field
        id_layout = QHBoxLayout()
        self.id_edit = QLineEdit(self.id_name)
        self.id_edit.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                color: white;
                padding: 5px;
                border: 1px solid #3d7ab8;
                border-radius: 3px;
            }
        """)
        self.save_id_btn = CustomButton("Save ID")
        self.save_id_btn.clicked.connect(self.save_id_changes)
        id_layout.addWidget(self.id_edit)
        id_layout.addWidget(self.save_id_btn)
        layout.addLayout(id_layout)
        
        # Filename
        filename_layout = QHBoxLayout()
        self.filename_label = QLabel(os.path.basename(self.image_path))
        self.filename_label.setStyleSheet("color: #888888; font-size: 10px;")
        filename_layout.addWidget(self.filename_label)
        layout.addLayout(filename_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.rename_btn = CustomButton("Rename")
        self.replace_btn = CustomButton("Replace")
        self.delete_btn = CustomButton("Delete")
        
        self.rename_btn.clicked.connect(self.rename_file)
        self.replace_btn.clicked.connect(self.replace_image)
        self.delete_btn.clicked.connect(self.delete_entry)
        
        button_layout.addWidget(self.rename_btn)
        button_layout.addWidget(self.replace_btn)
        button_layout.addWidget(self.delete_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #252526;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
    def load_image(self):
        if not self.image_loader:
            self.image_loader = ImageLoader(self.image_path)
            self.image_loader.image_loaded.connect(self.on_image_loaded)
            self.image_loader.start()
            
    def on_image_loaded(self, path, pixmap):
        if path == self.image_path:
            self.image_label.setPixmap(pixmap)
            self.image_loader = None
            
    def save_id_changes(self):
        new_id = self.id_edit.text().strip()
        if new_id and new_id != self.id_name:
            self.id_changed.emit(self.id_name, new_id)
            
    def rename_file(self):
        old_name = os.path.basename(self.image_path)
        new_name, ok = QInputDialog.getText(self, "Rename File", 
                                          "Enter new filename:",
                                          QLineEdit.Normal,
                                          old_name)
        if ok and new_name:
            self.file_renamed.emit(self.id_name, new_name)
            
    def replace_image(self):
        self.image_replaced.emit(self.id_name)
        
    def delete_entry(self):
        reply = QMessageBox.question(self, "Confirm Delete",
                                   f"Are you sure you want to delete {self.id_name}?",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.entry_deleted.emit(self.id_name)

class VirtualGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        self.items = []
        self.visible_cards = {}
        self.card_height = 300
        self.columns = 4
        self.setStyleSheet("background-color: #1e1e1e;")

    def add_item(self, id_name, image_path):
        self.items.append((id_name, image_path))
        self.update_total_size()

    def clear(self):
        self.items.clear()
        for card in self.visible_cards.values():
            card.deleteLater()
        self.visible_cards.clear()
        self.update_total_size()

    def update_total_size(self):
        total_rows = (len(self.items) + self.columns - 1) // self.columns
        total_height = total_rows * self.card_height
        self.setMinimumHeight(total_height)
        self.setMaximumHeight(total_height)

    def sizeHint(self):
        total_rows = (len(self.items) + self.columns - 1) // self.columns
        return QSize(800, total_rows * self.card_height)

    def update_visible_area(self, viewport_rect, callback_handler):
        viewport_top = viewport_rect.top()
        viewport_bottom = viewport_rect.bottom()
        
        # Calculate visible rows based on scroll position
        start_row = max(0, viewport_top // self.card_height - 1)  # Add one row buffer
        end_row = min(len(self.items) // self.columns + 1, 
                     viewport_bottom // self.card_height + 2)  # Add one row buffer
        
        visible_indices = set()
        for row in range(start_row, end_row):
            for col in range(self.columns):
                idx = row * self.columns + col
                if idx < len(self.items):
                    visible_indices.add(idx)

        # Remove non-visible cards
        for idx in list(self.visible_cards.keys()):
            if idx not in visible_indices:
                self.visible_cards[idx].deleteLater()
                del self.visible_cards[idx]

        # Add new visible cards
        for idx in visible_indices:
            if idx not in self.visible_cards and idx < len(self.items):
                id_name, image_path = self.items[idx]
                card = LazyImageCard(id_name, image_path)
                
                card.id_changed.connect(callback_handler.on_id_changed)
                card.image_replaced.connect(callback_handler.on_image_replaced)
                card.entry_deleted.connect(callback_handler.on_entry_deleted)
                card.file_renamed.connect(callback_handler.on_file_renamed)
                
                row = idx // self.columns
                col = idx % self.columns
                card_width = viewport_rect.width() // self.columns
                x = col * card_width
                y = row * self.card_height
                
                card.setGeometry(x, y, card_width, self.card_height)
                card.setParent(self)
                card.show()
                card.load_image()
                
                self.visible_cards[idx] = card

class DatabaseManagerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.face_db = {}
        self.current_file = None
        self.current_page = 1
        self.items_per_page = 100
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Face Database Manager")
        self.setMinimumSize(1024, 768)
        self.setWindowIcon(QIcon('icon.ico'))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)

        # Database selection
        db_layout = QHBoxLayout()
        self.db_path_label = QLabel("No database selected")
        select_db_btn = CustomButton("Select Database")
        select_db_btn.clicked.connect(self.select_database)
        db_layout.addWidget(self.db_path_label)
        db_layout.addWidget(select_db_btn)
        layout.addLayout(db_layout)

        # Search and display options
        options_layout = QHBoxLayout()
        
        # Search bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by ID...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                color: white;
                padding: 5px;
                border: 1px solid #3d7ab8;
                border-radius: 3px;
            }
        """)
        
        # Items per page selector
        self.items_per_page_combo = QComboBox()
        self.items_per_page_combo.addItems(['100', '200', '300'])
        self.items_per_page_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d;
                color: white;
                padding: 5px;
                border: 1px solid #3d7ab8;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        self.items_per_page_combo.currentTextChanged.connect(self.on_items_per_page_changed)
        
        options_layout.addWidget(self.search_input)
        options_layout.addWidget(QLabel("Items per page:"))
        options_layout.addWidget(self.items_per_page_combo)
        layout.addLayout(options_layout)

        # Pagination controls
        pagination_layout = QHBoxLayout()
        self.prev_page_btn = CustomButton("Previous")
        self.next_page_btn = CustomButton("Next")
        self.page_label = QLabel("Page 1")
        self.page_label.setStyleSheet("color: white;")
        
        self.prev_page_btn.clicked.connect(self.previous_page)
        self.next_page_btn.clicked.connect(self.next_page)
        
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.prev_page_btn)
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addWidget(self.next_page_btn)
        pagination_layout.addStretch()
        layout.addLayout(pagination_layout)

        # Create scroll container for virtual grid
        self.scroll_container = ScrollContainer()
        layout.addWidget(self.scroll_container)
        
        # Virtual grid setup
        self.virtual_grid = VirtualGrid()
        self.scroll_container.scroll_area.setWidget(self.virtual_grid)
        self.scroll_container.scroll_area.verticalScrollBar().valueChanged.connect(self.update_visible_cards)

        # Search functionality setup
        self.search_timer = QtCore.QTimer()
        self.search_timer.setSingleShot(True)
        self.search_timer.timeout.connect(self.filter_images)
        self.search_input.textChanged.connect(self.on_search_text_changed)

        # Main window styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
        """)

    def on_items_per_page_changed(self, value):
        self.items_per_page = int(value)
        self.current_page = 1
        self.filter_images()

    def previous_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.filter_images()

    def next_page(self):
        total_pages = self.get_total_pages()
        if self.current_page < total_pages:
            self.current_page += 1
            self.filter_images()

    def get_total_pages(self):
        filtered_items = [id_name for id_name in self.face_db.keys() 
                         if self.search_input.text().lower() in id_name.lower()]
        return max(1, (len(filtered_items) + self.items_per_page - 1) // self.items_per_page)

    def update_pagination_controls(self):
        total_pages = self.get_total_pages()
        self.page_label.setText(f"Page {self.current_page} of {total_pages}")
        self.prev_page_btn.setEnabled(self.current_page > 1)
        self.next_page_btn.setEnabled(self.current_page < total_pages)

    def select_database(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Database File", "", "Pickle Files (*.pkl)")
        
        if file_path:
            self.current_file = file_path
            self.db_path_label.setText(file_path)
            self.load_database(file_path)

    def load_database(self, file_path):
        progress = QProgressDialog("Loading database...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        self.load_thread = LoadDatabaseThread(file_path)
        self.load_thread.finished.connect(lambda db: self.on_database_loaded(db, progress))
        self.load_thread.start()

    def on_database_loaded(self, db, progress):
        self.face_db = db
        progress.close()
        self.current_page = 1
        self.filter_images()

    def on_search_text_changed(self):
        self.search_timer.start(300)

    def filter_images(self):
        self.virtual_grid.clear()
        search_text = self.search_input.text().lower()
        
        # Filter and sort items
        filtered_items = [(id_name, data) for id_name, data in self.face_db.items()
                         if search_text in id_name.lower()]
        filtered_items.sort(key=lambda x: x[0])  # Sort by ID
        
        # Calculate pagination
        start_idx = (self.current_page - 1) * self.items_per_page
        end_idx = start_idx + self.items_per_page
        page_items = filtered_items[start_idx:end_idx]
        
        # Add items to grid
        for id_name, data in page_items:
            self.virtual_grid.add_item(id_name, data['image_path'])
        
        self.update_pagination_controls()
        self.update_visible_cards()

    def update_visible_cards(self):
        viewport_rect = self.scroll_container.scroll_area.viewport().rect()
        viewport_rect.translate(0, self.scroll_container.scroll_area.verticalScrollBar().value())
        self.virtual_grid.update_visible_area(viewport_rect, self)

    def on_id_changed(self, old_id, new_id):
        if new_id in self.face_db:
            QMessageBox.warning(self, "Error", "ID already exists!")
            return
            
        self.face_db[new_id] = self.face_db.pop(old_id)
        self.save_database()
        self.filter_images()

    def on_file_renamed(self, id_name, new_filename):
        try:
            old_path = self.face_db[id_name]['image_path']
            new_path = os.path.join(os.path.dirname(old_path), new_filename)
            
            # Check if new filename already exists
            if os.path.exists(new_path):
                QMessageBox.warning(self, "Error", "File name already exists!")
                return
                
            # Rename the file
            shutil.move(old_path, new_path)
            
            # Update database
            self.face_db[id_name]['image_path'] = new_path
            self.save_database()
            self.filter_images()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error renaming file: {str(e)}")

    def on_image_replaced(self, id_name):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Replacement Image", "", 
                "Images (*.png *.jpg *.jpeg)")
            
            if file_path:
                # Verify the new image
                img = cv2.imread(file_path)
                if img is None:
                    raise Exception("Invalid image file")

                # Get the destination path (keeping original filename)
                dest_path = self.face_db[id_name]['image_path']
                
                # Replace the image
                shutil.copy2(file_path, dest_path)
                
                # Refresh the display
                self.filter_images()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error replacing image: {str(e)}")

    def on_entry_deleted(self, id_name):
        try:
            # Delete the image file
            image_path = self.face_db[id_name]['image_path']
            if os.path.exists(image_path):
                os.remove(image_path)
            
            # Remove from database
            del self.face_db[id_name]
            
            # Save changes
            self.save_database()
            
            # Refresh display
            self.filter_images()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error deleting entry: {str(e)}")

    def save_database(self):
        try:
            if not self.current_file:
                return
                
            backup_path = self.current_file + '.backup'
            
            # Create backup of current database
            if os.path.exists(self.current_file):
                shutil.copy2(self.current_file, backup_path)
            
            # Save the updated database
            with open(self.current_file, 'wb') as f:
                data = {
                    'embeddings': self.face_db,
                    'last_modified': datetime.now()
                }
                pickle.dump(data, f)
            
            # Remove backup if save was successful
            if os.path.exists(backup_path):
                os.remove(backup_path)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving database: {str(e)}")
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, self.current_file)
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the application?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.save_database()
            event.accept()
        else:
            event.ignore()

def main():
    app = QtWidgets.QApplication([])
    window = DatabaseManagerApp()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()