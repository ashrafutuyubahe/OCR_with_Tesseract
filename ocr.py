import sys
import cv2
import pytesseract
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QSlider, QFileDialog, QMessageBox, QTabWidget, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtWidgets import QScrollArea

# IMPORTANT FOR WINDOWS
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCRPreprocessor:
    """Handles all OCR preprocessing and extraction"""
    
    @staticmethod
    def preprocess_image(image, contrast=255, exposure=255, saturation=0, threshold=75):
        """Preprocess image for better OCR results"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check background color
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        white_pixel_count = np.sum(mask == 255)
        is_white_background = white_pixel_count > (mask.size / 2)
        
        inverted_image = cv2.bitwise_not(gray) if is_white_background else gray
        
        # Binarize
        binarized_image = cv2.convertScaleAbs(inverted_image, alpha=contrast/255, beta=0)
        binarized_image = cv2.convertScaleAbs(binarized_image, alpha=exposure/255, beta=0)
        binarized_image = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2BGR)
        binarized_image = cv2.cvtColor(binarized_image, cv2.COLOR_BGR2HSV)
        binarized_image[:, :, 1] = saturation
        binarized_image = cv2.cvtColor(binarized_image, cv2.COLOR_HSV2BGR)
        
        _, binary_image = cv2.threshold(binarized_image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image
    
    @staticmethod
    def extract_text(image):
        """Extract text using Tesseract"""
        try:
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            return f"Error extracting text: {str(e)}"


class ImageLabel(QLabel):
    """Custom label for ROI selection"""
    
    def __init__(self):
        super().__init__()
        self.roi_start = None
        self.roi_end = None
        self.drawing = False
        self.setStyleSheet("border: 2px solid gray;")
        self.setMinimumSize(400, 300)
    
    def mousePressEvent(self, event):
        self.roi_start = event.pos()
        self.drawing = True
    
    def mouseMoveEvent(self, event):
        if self.drawing and self.roi_start:
            self.roi_end = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event):
        self.roi_end = event.pos()
        self.drawing = False
    
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing and self.roi_start and self.roi_end:
            from PyQt5.QtGui import QPainter, QPen, QColor
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            rect = QRect(self.roi_start, self.roi_end)
            painter.drawRect(rect)
    
    def get_roi_rect(self):
        """Returns ROI rectangle normalized to 0-1"""
        if self.roi_start and self.roi_end:
            x1 = min(self.roi_start.x(), self.roi_end.x())
            y1 = min(self.roi_start.y(), self.roi_end.y())
            x2 = max(self.roi_start.x(), self.roi_end.x())
            y2 = max(self.roi_start.y(), self.roi_end.y())
            return (x1, y1, x2, y2)
        return None


class OCRScannerGUI(QMainWindow):
    """Main GUI Application for Text Scanner"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Without ML - Printed Text Scanner")
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_image = None
        self.processed_image = None
        self.camera = None
        self.camera_active = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        
        # Left panel - Image display and controls
        left_panel = QVBoxLayout()
        
        # Tabs for different input methods
        self.tabs = QTabWidget()
        
        # Tab 1: Load Image
        load_tab = QWidget()
        load_layout = QVBoxLayout()
        self.image_label = ImageLabel()
        load_layout.addWidget(QLabel("Select ROI (click and drag on image):"))
        load_layout.addWidget(self.image_label)
        load_tab.setLayout(load_layout)
        
        # Tab 2: Camera
        camera_tab = QWidget()
        camera_layout = QVBoxLayout()
        self.camera_label = ImageLabel()
        camera_layout.addWidget(self.camera_label)
        camera_tab.setLayout(camera_layout)
        
        self.tabs.addTab(load_tab, "Load Image")
        self.tabs.addTab(camera_tab, "Live Camera")
        
        left_panel.addWidget(self.tabs)
        
        # Button controls
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_btn)
        
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self.toggle_camera)
        button_layout.addWidget(self.camera_btn)
        
        self.scan_btn = QPushButton("Run OCR")
        self.scan_btn.clicked.connect(self.run_ocr)
        self.scan_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.scan_btn)
        
        left_panel.addLayout(button_layout)
        
        # Preprocessing controls
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(QLabel("Preprocessing Parameters:"))
        
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 255)
        self.contrast_slider.setValue(255)
        self.contrast_label = QLabel("255")
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_label)
        controls_layout.addLayout(contrast_layout)
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(75)
        self.threshold_label = QLabel("75")
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)
        controls_layout.addLayout(threshold_layout)
        
        self.contrast_slider.valueChanged.connect(self.update_contrast_label)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        
        left_panel.addLayout(controls_layout)
        
        # Right panel - Text output
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("Extracted Text:"))
        
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        font = QFont("Courier")
        font.setPointSize(10)
        self.text_output.setFont(font)
        right_panel.addWidget(self.text_output)
        
        # Status label
        self.status_label = QLabel("Ready")
        right_panel.addWidget(self.status_label)
        
        # Copy button
        copy_btn = QPushButton("Copy Text")
        copy_btn.clicked.connect(self.copy_text)
        right_panel.addWidget(copy_btn)
        
        # Combine panels
        main_layout.addLayout(left_panel, 60)
        main_layout.addLayout(right_panel, 40)
        
        central_widget.setLayout(main_layout)
    
    def load_image(self):
        """Load image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.bmp *.tiff)"
        )
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image, self.image_label)
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_active:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera_active = True
                self.camera_btn.setText("Stop Camera")
                self.timer.start(30)
                self.status_label.setText("Camera active")
            else:
                QMessageBox.warning(self, "Error", "Could not open camera")
        else:
            self.camera_active = False
            self.camera_btn.setText("Start Camera")
            self.timer.stop()
            if self.camera:
                self.camera.release()
            self.status_label.setText("Camera stopped")
    
    def update_camera(self):
        """Update camera frame"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.current_image = frame
                self.display_image(frame, self.camera_label)
    
    def display_image(self, cv_image, label):
        """Convert and display OpenCV image to PyQt5 label"""
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaledToWidth(400, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
    
    def run_ocr(self):
        """Run OCR on the current image"""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "Please load an image or start camera first")
            return
        
        self.status_label.setText("Processing... Please wait")
        QApplication.processEvents()
        
        # Get ROI if selected
        current_label = (self.image_label if self.tabs.currentIndex() == 0 
                        else self.camera_label)
        roi_rect = current_label.get_roi_rect()
        
        image_to_process = self.current_image
        if roi_rect:
            h, w = self.current_image.shape[:2]
            x1, y1, x2, y2 = roi_rect
            image_to_process = self.current_image[y1:y2, x1:x2]
        
        # Preprocess image
        contrast = self.contrast_slider.value()
        threshold = self.threshold_slider.value()
        self.processed_image = OCRPreprocessor.preprocess_image(
            image_to_process, contrast=contrast, threshold=threshold
        )
        
        # Extract text
        text = OCRPreprocessor.extract_text(self.processed_image)
        self.text_output.setText(text)
        
        # Display processed image
        self.display_image(self.processed_image, current_label)
        
        self.status_label.setText(f"OCR Complete - {len(text.split())} words found")
    
    def copy_text(self):
        """Copy extracted text to clipboard"""
        text = self.text_output.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.setText("Text copied to clipboard!")
        else:
            QMessageBox.information(self, "Info", "No text to copy")
    
    def update_contrast_label(self):
        """Update contrast label value"""
        self.contrast_label.setText(str(self.contrast_slider.value()))
    
    def update_threshold_label(self):
        """Update threshold label value"""
        self.threshold_label.setText(str(self.threshold_slider.value()))
    
    def closeEvent(self, event):
        """Clean up resources on close"""
        if self.camera:
            self.camera.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCRScannerGUI()
    window.show()
    sys.exit(app.exec_())