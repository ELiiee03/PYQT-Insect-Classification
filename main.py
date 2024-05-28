import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QWidget, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor
from PyQt5.QtCore import Qt
from resnet50_predict import predict_image

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Insect Classification')
        self.setGeometry(100, 100, 1200, 800)

        # Set the background color for the main window
        self.setStyleSheet("background-color: #c1bcc5")

        # Main layout
        self.main_layout = QHBoxLayout()

        # Left side: Image display
        self.left_layout = QVBoxLayout()
        self.label_image = QLabel()
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setMinimumSize(800, 400)
        self.label_image.setStyleSheet("background-color: f48d03; padding: 10px;")
        self.left_layout.addWidget(self.label_image)
        self.btn_browse = QPushButton('Upload Insect Image')
        self.btn_browse.setStyleSheet(
            "background-color: #f48d03; font-size: 25px; font-weight: bold; color: black; padding: 20px; border-radius: 20px;"
        )
        self.btn_browse.clicked.connect(self.show_image)
        self.left_layout.addWidget(self.btn_browse)

        # Right side: Classification display
        self.right_layout = QVBoxLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: black;")
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.right_layout.addWidget(self.scroll_area)
        self.btn_exit = QPushButton('Exit')
        self.btn_exit.setStyleSheet(
            "background-color: #f48d03; font-size: 25px; font-weight: bold; color: black; padding: 20px; border-radius: 20px;"
        )
        self.btn_exit.clicked.connect(self.close)
        self.right_layout.addWidget(self.btn_exit)

        self.main_layout.addLayout(self.left_layout)
        self.main_layout.addLayout(self.right_layout)

        # Create the main widget and set the layout
        self.main_widget = QWidget()
        self.main_widget.setLayout(self.main_layout)

        # Create the overall layout and add the main widget
        self.overall_layout = QVBoxLayout()
        self.overall_layout.addWidget(self.main_widget)

        # Set the central widget
        central_widget = QWidget()
        central_widget.setLayout(self.overall_layout)
        self.setCentralWidget(central_widget)

    def show_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', os.getcwd(), 'Image Files (*.jpg *.png)'
        )
        if file_path:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
            self.label_image.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self.label_image.size(), Qt.AspectRatioMode.KeepAspectRatio
                )
            )
            self.predict_image(file_path)

    def predict_image(self, img_path):
        predictions = predict_image(img_path)
        self.display_classification_results(predictions)

    def display_classification_results(self, predictions):
        # Clear the existing classification results
        for i in reversed(range(self.scroll_layout.count())):
            self.scroll_layout.itemAt(i).widget().setParent(None)

        # Display the new classification results
        for prediction in predictions:
            label = QLabel(f"Insect Classification: {prediction}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
            self.scroll_layout.addWidget(label)


if __name__ == '__main__':
    app = QApplication([])

    # Set the application palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    app.setPalette(palette)

    viewer = ImageViewer()
    viewer.show()
    app.exec_()
